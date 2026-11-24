"""Versioned, portable workload specifications; no data or GPU downloads."""
from dataclasses import dataclass
from importlib.resources import files
import hashlib
import json
import math
from pathlib import Path

from .contract import DTYPE_TOLERANCES, KernelConfig, Workload
from .search import CandidateSpec, TaskSpec


BUNDLED_SUITES = ('operator-search-v1', 'operator-generalization-v2')


@dataclass(frozen=True)
class Suite:
    _document_json: str
    sha256: str
    tasks: tuple[TaskSpec, ...]
    candidates: tuple[CandidateSpec, ...]
    _splits: tuple[tuple[str, tuple[int, ...]], ...]

    @property
    def document(self):
        return json.loads(self._document_json)

    @property
    def splits(self):
        return dict(self._splits)


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def load_suite(path=None):
    identifier = 'operator-search-v1' if path is None else str(path)
    raw = (files('drivebench_tuning').joinpath(f'suites/{identifier}.json').read_bytes()
           if identifier in BUNDLED_SUITES else Path(path).read_bytes())
    doc = json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_strict_object)
    if not isinstance(doc, dict):
        raise ValueError("suite must be a JSON object")
    # Also rejects finite-looking exponents which overflow float (e.g. 1e999).
    json.dumps(doc, allow_nan=False)
    if type(doc.get('schema_version')) is not int or doc['schema_version'] != 1 or not isinstance(doc.get('suite_id'), str) or not doc['suite_id']:
        raise ValueError('suite requires schema_version 1 and a nonempty suite_id')
    tasks, candidates, ids, groups = [], [], set(), {}
    splits = {name: [] for name in ('train', 'validation', 'test')}
    for row in doc['tasks']:
        if set(row) != {'id', 'split', 'workload'} or row['split'] not in splits:
            raise ValueError('task needs id, split and workload; unknown split')
        if row['id'] in ids:
            raise ValueError('duplicate task ID')
        ids.add(row['id'])
        w = Workload(**row['workload'])
        # A new random seed or dtype does not make a held-out shape.
        # Group count changes do not create a held-out tensor shape either.
        group = (w.operator, w.shape)
        if group in groups and groups[group] != row['split']:
            raise ValueError('shape family overlaps train/validation/test splits')
        groups[group] = row['split']
        features = (float(w.operator == 'groupnorm_silu'),
                    {'float16': 0., 'bfloat16': .5, 'float32': 1.}[w.dtype],
                    len(w.shape) / 4., *(math.log2(d) / 32. for d in w.shape),
                    *((0.,) * (4 - len(w.shape))), math.log2(w.num_groups) / 16.)
        tasks.append(TaskSpec(row['id'], features, {'workload': row['workload'], 'split': row['split']}))
        splits[row['split']].append(len(tasks)-1)
    for row in doc['candidates']:
        if set(row) != {'id', 'config'}:
            raise ValueError('candidate needs id and config')
        config = row['config']
        if config.get('implementation') == 'compiled':
            if set(config) != {'implementation'}:
                raise ValueError('compiled candidate takes no launch settings')
        elif config.get('implementation') == 'triton':
            if set(config) != {'implementation', 'block_size', 'num_warps'}:
                raise ValueError('triton candidate needs block_size and num_warps')
            KernelConfig(config['block_size'], config['num_warps'])
        else:
            raise ValueError('bundled operator adapter supports compiled and triton candidates')
        candidates.append(CandidateSpec(row['id'], config))
    if not tasks or not candidates or any(not v for v in splits.values()):
        raise ValueError('nonempty train, validation, test and candidates required')
    if len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError('duplicate candidate ID')
    if len({json.dumps(c.to_dict()['config'], sort_keys=True) for c in candidates}) != len(candidates):
        raise ValueError('duplicate candidate configuration')
    budget = doc['default_trial_budget']
    if type(budget) is not int or not 0 < budget < len(candidates):
        raise ValueError('suite search budget must be positive and smaller than the action count')
    weight = doc['default_memory_weight']
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError('memory weight must be finite and nonnegative')
    if 'quality_tolerances' in doc and doc['quality_tolerances'] != DTYPE_TOLERANCES:
        raise ValueError('suite numerical tolerances differ from the live evaluator')
    return Suite(json.dumps(doc, allow_nan=False), hashlib.sha256(raw).hexdigest(), tuple(tasks), tuple(candidates),
                 tuple((k, tuple(v)) for k, v in splits.items()))


def load_bound_suite(document, sha256):
    """Resolve only an exact packaged suite; embedded JSON cannot redefine it."""
    identifier = document.get('suite_id') if isinstance(document, dict) else None
    if identifier not in BUNDLED_SUITES:
        raise ValueError('Recorded suite is not a supported bundled suite')
    suite = load_suite(identifier)
    if suite.sha256 != sha256 or suite.document != document:
        raise ValueError('Recorded suite document or SHA256 differs from packaged suite')
    return suite


def suite_task_indices(suite, splits=None):
    """Canonical task subset; selecting a split never changes the suite hash."""
    if splits is None:
        return tuple(range(len(suite.tasks)))
    if (not isinstance(splits, (list, tuple)) or not splits
            or len(set(splits)) != len(splits) or any(s not in suite.splits for s in splits)):
        raise ValueError('Task splits must be a nonempty unique train/validation/test list')
    wanted = {index for split in splits for index in suite.splits[split]}
    return tuple(index for index in range(len(suite.tasks)) if index in wanted)
