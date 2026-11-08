from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts import probe_compiler_operator as probe


def test_primary_integrated_config_is_accepted():
    config, generation, options = probe.inspect_config('configs/study_compiler_operator.yaml')
    assert generation.dtype == 'fp16' and options.compile_denoiser
    assert options.max_batch_size == 1 and options.feature_cache_interval == 1
    assert config['operator_probe']['request_index'] == 28


@pytest.mark.parametrize('change', [
    {'generation': {'steps': 49}}, {'generation': {'dtype': 'bf16'}},
    {'optimizations': {'conditioning_cache_bytes': 10}},
    {'optimizations': {'max_batch_size': 2}},
    {'model': {'compiler_runtime': None}},
    {'operator_probe': {'request_index': 32}}, {'operator_probe': {'request_index': True}},
    {'numerical_validation': 'proof.json'}, {'benchmark': {'headline': True}},
])
def test_changed_contract_is_rejected(tmp_path, change):
    config, _, _ = probe.inspect_config('configs/study_compiler_operator.yaml')
    for key, value in change.items():
        if isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    path = tmp_path / 'config.yaml'; path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        probe.inspect_config(path)


def test_capture_preserves_exact_objects_and_restores_compiled_callable():
    values = [object() for _ in range(4)]
    original = lambda *args: pytest.fail('Must stop before actual denoiser call')
    backend = SimpleNamespace(_denoiser=original)
    backend.generate_batch = lambda *args: backend._denoiser(*values)
    result = probe.capture_first_input(backend, None, {'trajectories': [1], 'seeds': [2]}, None)
    assert list(result.values()) == values and backend._denoiser is original


def test_capture_restores_callable_on_unrelated_failure():
    original = object(); backend = SimpleNamespace(_denoiser=original)
    def fail(*args):
        raise RuntimeError('conditioning failed')
    backend.generate_batch = fail
    with pytest.raises(RuntimeError, match='conditioning failed'):
        probe.capture_first_input(backend, None, {'trajectories': [1], 'seeds': [2]}, None)
    assert backend._denoiser is original


def test_comparison_reports_nonzero_error_without_changing_zero_threshold():
    left = np.zeros((2, 3, 4), dtype=np.float32); right = left.copy(); right[0, 0, 0] = 2
    result = probe.compare_arrays(left, right)
    assert not result['passed_exact'] and result['max_absolute_error'] == 2
    assert result['rmse'] == pytest.approx(np.sqrt(4 / 24))
    assert result['different_values'] == 1 and result['atol'] == result['rtol'] == 0
    assert probe.compare_arrays(left, left)['passed_exact']


@pytest.mark.parametrize('right', [np.zeros((1, 3, 4)), np.full((2, 3, 4), np.nan), np.zeros((2, 3, 4), dtype=int)])
def test_invalid_outputs_fail_closed(right):
    with pytest.raises(ValueError):
        probe.compare_arrays(np.zeros((2, 3, 4), dtype=np.float32), right)


def test_input_array_hash_detects_value_change():
    a = np.zeros((2, 3), dtype=np.float16)
    before = probe.array_hash(a)
    a[0, 0] = 1
    assert probe.array_hash(a) != before


def test_equal_values_with_changed_dtype_are_not_an_exact_contract_pass():
    result = probe.compare_arrays(np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float16))
    assert result['max_absolute_error'] == 0 and result['rmse'] == 0
    assert not result['dtype_equal'] and not result['passed_exact']
