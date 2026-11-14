"""Mocked runtime observations; no CUDA execution or forced attention mode."""
import json
import sys
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from drivebench_tuning import vista_agent


def mocked_runtime(monkeypatch, *, cudnn_sdp=False):
    flags = {'flash_sdp_enabled': True, 'mem_efficient_sdp_enabled': True,
             'math_sdp_enabled': True, 'cudnn_sdp_enabled': False}
    backend = NS(matmul=NS(allow_tf32=False))
    for name in ('flash_sdp_enabled', 'mem_efficient_sdp_enabled', 'math_sdp_enabled'):
        setattr(backend, name, lambda name=name: flags[name])
    if cudnn_sdp:
        backend.cudnn_sdp_enabled = lambda: flags['cudnn_sdp_enabled']
    torch = NS(cuda=NS(is_available=lambda: True, device_count=lambda: 1,
                      get_device_name=lambda index: 'Mocked A100'),
               version=NS(cuda='11.8'),
               backends=NS(cuda=backend, cudnn=NS(version=lambda: 8700,
                   deterministic=False, benchmark=False, allow_tf32=True)),
               are_deterministic_algorithms_enabled=lambda: False,
               is_deterministic_algorithms_warn_only_enabled=lambda: False,
               get_float32_matmul_precision=lambda: 'highest')
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(vista_agent.importlib.metadata, 'version', lambda name: 'mocked-' + name)
    monkeypatch.setattr(vista_agent.subprocess, 'run', lambda *a, **kw: NS(stdout='GPU-mocked, 580.178.04\n'))
    return flags, backend


def test_torch20_runtime_records_absent_newer_cudnn_api_without_changing_flags(monkeypatch):
    flags, _ = mocked_runtime(monkeypatch)
    before = dict(flags)
    runtime = vista_agent.probe_runtime()
    assert runtime['sdp'] == {'flash_sdp_enabled': True, 'mem_efficient_sdp_enabled': True,
        'math_sdp_enabled': True, 'cudnn_sdp_enabled_available': False, 'cudnn_sdp_enabled': None}
    assert flags == before
    assert runtime == vista_agent.probe_runtime()


@pytest.mark.parametrize('flag', ['flash_sdp_enabled', 'mem_efficient_sdp_enabled', 'math_sdp_enabled', 'cudnn_sdp_enabled'])
def test_observed_algorithm_flag_drift_changes_runtime_identity(monkeypatch, flag):
    flags, _ = mocked_runtime(monkeypatch, cudnn_sdp=True)
    before = vista_agent.probe_runtime()
    flags[flag] = not flags[flag]
    after = vista_agent.probe_runtime()
    assert after['sdp'][flag] is flags[flag]
    assert after != before and vista_agent.digest(after) != vista_agent.digest(before)


def test_cudnn_api_availability_is_part_of_identity(monkeypatch):
    _, backend = mocked_runtime(monkeypatch)
    before = vista_agent.probe_runtime()
    backend.cudnn_sdp_enabled = lambda: False
    after = vista_agent.probe_runtime()
    assert after['sdp']['cudnn_sdp_enabled_available'] is True
    assert after['sdp']['cudnn_sdp_enabled'] is False
    assert after != before


@pytest.mark.parametrize('kind', ['missing_required', 'nonboolean', 'invalid_optional'])
def test_unobservable_algorithm_settings_fail_closed(monkeypatch, kind):
    flags, backend = mocked_runtime(monkeypatch)
    if kind == 'missing_required':
        del backend.math_sdp_enabled
    elif kind == 'nonboolean':
        flags['flash_sdp_enabled'] = 1
    else:
        backend.cudnn_sdp_enabled = False
    with pytest.raises(ValueError, match='SDPA'):
        vista_agent.probe_runtime()


def test_synthetic_identity_is_unchanged_without_torch_or_subprocess(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    monkeypatch.setattr(vista_agent.subprocess, 'run', lambda *a, **kw: pytest.fail('synthetic queried GPU'))
    assert vista_agent.probe_runtime(synthetic=True) == {'kind': 'synthetic', 'numpy': np.__version__, 'gpu': None}


@pytest.mark.parametrize('stage', ['before_start', 'after_generation'])
def test_batch_boundary_rejects_algorithm_drift_and_keeps_partial_arrays(tmp_path, monkeypatch, stage):
    from drivebench.engine import InferenceEngine
    flags, _ = mocked_runtime(monkeypatch)
    probe = vista_agent.probe_runtime
    expected = probe()
    # Use the actual synthetic inference worker with an explicitly mocked real
    # runtime observation. This verifies the boundary, not GPU computation.
    monkeypatch.setattr(vista_agent, 'probe_runtime', lambda **kw: probe(synthetic=False))
    settings = vista_agent.reference_settings(synthetic=True)
    request = vista_agent.BatchRequest('one', 'scene', 11, 0,
        np.zeros((1, 3, 8, 8), dtype=np.float32), np.zeros(8, dtype=np.float32))
    if stage == 'before_start':
        flags['mem_efficient_sdp_enabled'] = False
    else:
        generate = InferenceEngine.generate
        def changed(self, *args, **kwargs):
            result = generate(self, *args, **kwargs)
            flags['mem_efficient_sdp_enabled'] = False
            return result
        monkeypatch.setattr(InferenceEngine, 'generate', changed)
    output = tmp_path/'batch'
    with pytest.raises(ValueError, match='runtime drift'):
        vista_agent.run_batch(output, settings=settings, model={'backend': 'synthetic'}, requests=[request],
            cohort_id='mocked-runtime-boundary', expected_runtime=expected, source_files={},
            deadline_unix=time.time()+60, synthetic=True, minimum_free_bytes=0)
    assert not (output/'completed.json').exists()
    if stage == 'before_start':
        assert not output.exists()
    else:
        assert (output/'00000.npy').is_file()
        assert json.loads((output/'failure.json').read_text())['finished_warmups'] == 1
