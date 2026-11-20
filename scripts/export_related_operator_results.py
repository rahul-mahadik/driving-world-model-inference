#!/usr/bin/env python3
"""Publish derived policy/confirmation records against a sanitized operator table.

All GPU measurements and policy calls happened before this offline export.
Only publication bindings, hardware identifiers and private error text change.
Raw source files are hash-checked before and after export and remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent / 'extensions/kernel_tuning'))
from drivebench_tuning.benchmark import utc_now
from drivebench_tuning.replay import MeasuredTable


def _load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


public = _load_script('operator_table_public_export', 'export_operator_table.py')
plots = _load_script('operator_related_plot_validation', 'plot_operator_search.py')
_CONFIRMATION_FILES = ('plan.json', 'result.json', 'runtime.json', 'trials.jsonl', 'completed.json')
_ERROR_KEYS = {'error', 'validation_error', 'close_error'}
_IDENTITY_KEYS = {'trial_id', 'source_trial_id', 'task_id', 'candidate_id', 'input_digest'}


def canonical_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def rebind(value, mapping, counts):
    """Rebind exact digests; redact execution text and pseudonymize API IDs."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in _ERROR_KEYS and item is not None:
                if isinstance(item, dict):
                    result[key] = rebind(item, mapping, counts)
                    if 'message' in item:
                        result[key]['message'] = public._ERROR_TEXT
                        result[key]['message_redacted'] = True
                        counts['error_messages'] += 1
                elif isinstance(item, str):
                    result[key] = public._ERROR_TEXT
                    counts['error_messages'] += 1
                else:
                    result[key] = rebind(item, mapping, counts)
            elif key == 'response_id' and isinstance(item, str):
                result[key] = 'response-sha256-' + hashlib.sha256(item.encode()).hexdigest()
                counts['provider_response_ids'] += 1
            else:
                result[key] = rebind(item, mapping, counts)
        return result
    if isinstance(value, list):
        return [rebind(item, mapping, counts) for item in value]
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


def protected_values(value, prefix=()):
    """Record every numeric value and measurement/input/task identifier."""
    found = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = prefix + (key,)
            if key in _IDENTITY_KEYS:
                found[path] = (type(item).__name__, item)
            found.update(protected_values(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(protected_values(item, prefix + (index,)))
    elif type(value) in (bool, int, float):
        found[prefix] = (type(value).__name__, value)
    return found


def unchanged(source, derived):
    source_values, derived_values = protected_values(source), protected_values(derived)
    if any(derived_values.get(path) != value for path, value in source_values.items()):
        raise ValueError('A numerical field or measurement identifier changed during export')


def _read_json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _verify_table_derivation(source, derived):
    metadata = derived._document.get('public_export', {})
    if (metadata.get('source_table_sha256') != source.sha256
            or metadata.get('source_cohort_id') != source.cohort_id
            or metadata.get('measurements_rerun') is not False):
        raise ValueError('Public table is not bound to this raw table')
    runtime, identities = public.public_runtime(source._document['runtime'])
    if runtime != derived._document['runtime']:
        raise ValueError('Public runtime is not the pseudonymized source runtime')
    expected = public.transform(source._document['rows'], source.cohort_id, derived.cohort_id,
                                redactions=[0])
    if expected != derived._document['rows'] or source._document['cells'] != derived._document['cells']:
        raise ValueError('Public table measurement values differ from source')
    return identities


def export_related(source_table, public_table, output, *, policy_reports=(), confirmation_dirs=()):
    source_table, public_table = (Path(p).resolve(strict=True) for p in (source_table, public_table))
    policies = [Path(p).resolve(strict=True) for p in policy_reports]
    confirmations = [Path(p).resolve(strict=True) for p in confirmation_dirs]
    output = Path(output).resolve()
    if not policies and not confirmations:
        raise ValueError('At least one policy report or confirmation directory required')
    if (output.exists() or any(output.is_relative_to(p) for p in confirmations)
            or output in {source_table.parent, public_table.parent}):
        raise ValueError('A new output directory outside raw confirmation directories is required')
    before = {source_table: public.sha(source_table), public_table: public.sha(public_table)}
    for path in policies:
        before[path] = public.sha(path)
    for directory in confirmations:
        for name in _CONFIRMATION_FILES:
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('Confirmation artifacts must be regular files')
            before[path] = public.sha(path)
    source = MeasuredTable.load(source_table)
    derived_table = MeasuredTable.load(public_table)
    raw_gpu_ids = _verify_table_derivation(source, derived_table)
    # Validate source science before creating any publication output.
    source_curves, _ = plots.policy_curves(policies, source)
    source_points, _ = plots.confirmation_points(confirmations, source)
    mapping = {source.sha256: derived_table.sha256, source.cohort_id: derived_table.cohort_id}
    output.mkdir(parents=True, exist_ok=False)
    artifacts, selection_bindings = [], []
    counts = {'error_messages': 0, 'provider_response_ids': 0}
    derived_policies, derived_confirmations = [], []

    def record(source_path, derived_path):
        artifacts.append({'artifact': derived_path.relative_to(output).as_posix(),
                          'source_sha256': before[source_path], 'derived_sha256': public.sha(derived_path)})

    def stamp(source_sha):
        return {'kind': 'sanitized_copy_of_recorded_results', 'source_artifact_sha256': source_sha,
                'measurements_rerun': False, 'policy_calls_rerun': False,
                'numeric_fields_and_trial_ids_unchanged': True,
                'publication_bindings_updated': True, 'provenance_receipt': 'export-receipt.json'}

    for index, path in enumerate(policies):
        document, _ = plots.read_json(path)
        transformed = rebind(document, mapping, counts)
        unchanged(document, transformed)
        transformed['public_export'] = stamp(before[path])
        target = output / f'policy-{index:03d}.json'
        public.write_json(target, transformed)
        mapping[before[path]] = public.sha(target)
        record(path, target)
        derived_policies.append(target)

    known_selections = {}
    for directory in confirmations:
        documents = {name: (_read_json_lines(directory / name) if name.endswith('.jsonl')
                            else plots.read_json(directory / name)[0]) for name in _CONFIRMATION_FILES}
        plan, report = documents['plan.json'], documents['result.json']
        original_selection = plan['selection_document']
        original_selection_sha = plan['selection_sha256']
        if (not isinstance(original_selection_sha, str)
                or not re.fullmatch(r'[0-9a-f]{64}', original_selection_sha)):
            raise ValueError('Recorded selection digest must be SHA256')
        canonical_source = canonical_bytes(original_selection)
        if original_selection_sha in known_selections:
            if known_selections[original_selection_sha]['source_canonical_bytes'] != canonical_source:
                raise ValueError('One source selection digest refers to different embedded documents')
            selection_info = known_selections[original_selection_sha]
        else:
            transformed_selection = rebind(original_selection, mapping, counts)
            unchanged(original_selection, transformed_selection)
            selection_bytes = canonical_bytes(transformed_selection)
            new_selection_sha = hashlib.sha256(selection_bytes).hexdigest()
            selections_directory = output / 'selections'
            selections_directory.mkdir(exist_ok=True)
            target = selections_directory / ('selection-' + new_selection_sha + '.json')
            with target.open('xb') as stream:
                stream.write(selection_bytes)
            selection_info = {'source_canonical_bytes': canonical_source,
                              'derived_sha256': new_selection_sha,
                              'artifact': target.relative_to(output).as_posix()}
            known_selections[original_selection_sha] = selection_info
            selection_bindings.append({'source_selection_sha256_as_recorded': original_selection_sha,
                'source_embedded_document_canonical_sha256': hashlib.sha256(canonical_source).hexdigest(),
                'source_serialization_note': 'Original selection bytes are not embedded; source plan hash binds the exact embedded document and recorded digest.',
                'derived_selection_sha256': new_selection_sha, 'artifact': selection_info['artifact']})
        mapping[original_selection_sha] = selection_info['derived_sha256']
        target_dir = output / ('confirmation-' + str(report['repeat_index']))
        target_dir.mkdir(exist_ok=False)
        for name in _CONFIRMATION_FILES:
            document = documents[name]
            if name == 'runtime.json':
                transformed, _ = public.public_runtime(document)
            else:
                transformed = rebind(document, mapping, counts)
            unchanged(document, transformed)
            if name in ('plan.json', 'result.json', 'completed.json'):
                transformed['public_export'] = stamp(before[directory / name])
                if name == 'plan.json':
                    transformed['public_export']['selection_artifact_from_export_root'] = selection_info['artifact']
            if name == 'plan.json':
                selection_file = output / selection_info['artifact']
                if (transformed['selection_sha256'] != public.sha(selection_file)
                        or canonical_bytes(transformed['selection_document']) != selection_file.read_bytes()):
                    raise ValueError('Derived selection bytes do not match confirmation plan')
            if name == 'completed.json' and 'artifact_sha256' in transformed:
                if set(transformed['artifact_sha256']) - set(_CONFIRMATION_FILES):
                    raise ValueError('Unknown completion artifact; refusing external path lookup')
                transformed['artifact_sha256'] = {key: public.sha(target_dir / key)
                    for key in transformed['artifact_sha256'] if key != 'completed.json'}
            target = target_dir / name
            if name.endswith('.jsonl'):
                with target.open('x') as stream:
                    for row in transformed:
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            else:
                public.write_json(target, transformed)
            record(directory / name, target)
        derived_confirmations.append(target_dir)

    curves, _ = plots.policy_curves(derived_policies, derived_table)
    points, _ = plots.confirmation_points(derived_confirmations, derived_table)
    if curves != source_curves or points != source_points:
        raise ValueError('Derived policy or confirmation analysis changed')
    for path in output.rglob('*'):
        if path.is_file():
            text = path.read_text().lower()
            if any(identity in text or identity.replace('-', '') in text for identity in raw_gpu_ids):
                raise ValueError('Unredacted hardware identifier in derived related result')
    if any(public.sha(path) != digest for path, digest in before.items()):
        raise ValueError('Raw source or public table changed during related export')
    receipt = {'schema_version': 1, 'kind': 'public_related_operator_result_export', 'completed': True,
               'exported_at': utc_now(), 'exporter_sha256': public.sha(__file__),
               'source_table_sha256': source.sha256, 'derived_table_sha256': derived_table.sha256,
               'source_cohort_id': source.cohort_id, 'derived_cohort_id': derived_table.cohort_id,
               'source_artifact_hashes_unchanged': True, 'numeric_fields_and_trial_ids_unchanged': True,
               'measurements_rerun': False, 'policy_calls_rerun': False,
               'source_and_derived_plot_validation_passed': True,
               'policy_report_count': len(policies), 'confirmation_repeat_count': len(confirmations),
               'redactions': counts, 'selection_bindings': selection_bindings, 'artifacts': artifacts,
               'policy_report_artifacts': [p.relative_to(output).as_posix() for p in derived_policies],
               'confirmation_artifacts': [p.relative_to(output).as_posix() for p in derived_confirmations]}
    public.write_json(output / 'export-receipt.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-table', required=True, type=Path)
    parser.add_argument('--public-table', required=True, type=Path)
    parser.add_argument('--policy-reports', nargs='*', default=[], type=Path)
    parser.add_argument('--confirmation-dirs', nargs='*', default=[], type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = export_related(args.source_table, args.public_table, args.output,
                             policy_reports=args.policy_reports, confirmation_dirs=args.confirmation_dirs)
    print(json.dumps({'completed': True, 'measurements_rerun': False, 'policy_calls_rerun': False,
                      'policy_reports': receipt['policy_report_count'],
                      'confirmation_repeats': receipt['confirmation_repeat_count'],
                      'receipt_sha256': public.sha(args.output / 'export-receipt.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
