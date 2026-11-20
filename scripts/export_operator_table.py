#!/usr/bin/env python3
"""Export a derived public copy of a measured operator table and its repeats.

Only hardware identifiers and execution-error text are redacted. Measurements,
trial IDs, input hashes, sample arrays and correctness outcomes are preserved.
All raw input artifacts remain unchanged. This command does not use a GPU or
contact a provider, and the exported copy is not a new measurement cohort.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'extensions/kernel_tuning'))
from drivebench_tuning import table_collection as collection
from drivebench_tuning.benchmark import utc_now

_FILES = ('plan.json', 'runtime.json', 'rows.json', 'trials.jsonl', 'completed.json')
_ERROR_TEXT = '[redacted: execution error text may contain host paths or provider identifiers]'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def normalized_gpu_uuid(value):
    if not isinstance(value, str):
        raise ValueError('GPU UUID must be a string')
    raw = value[4:] if value.lower().startswith('gpu-') else value
    try:
        return str(uuid.UUID(raw))
    except (ValueError, AttributeError):
        raise ValueError('Unrecognized physical GPU UUID; refuse a guessed redaction') from None


def gpu_pseudonym(value):
    normalized = normalized_gpu_uuid(value)
    digest = hashlib.sha256(('drivebench-public-gpu-v1:' + normalized).encode()).hexdigest()
    return 'gpu-sha256-' + digest


def public_runtime(runtime):
    """Preserve driver versions and map every inventory UUID consistently."""
    result = deepcopy(runtime)
    raw_id = result['gpu_uuid']
    identities = {normalized_gpu_uuid(raw_id)}
    result['gpu_uuid'] = gpu_pseudonym(raw_id)
    inventory = []
    for line in result['driver_inventory']:
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 2 or not re.fullmatch(r'[0-9]+(?:\.[0-9]+)+', parts[1]):
            raise ValueError('Unrecognized driver inventory; refuse a guessed redaction')
        identities.add(normalized_gpu_uuid(parts[0]))
        inventory.append(gpu_pseudonym(parts[0]) + ', ' + parts[1])
    if not inventory:
        raise ValueError('Missing driver inventory')
    result['driver_inventory'] = inventory
    result.pop('cohort_id')
    result['cohort_id'] = collection._digest(result)
    return result, identities


def transform(value, old_cohort, new_cohort, *, redactions):
    """Change exact cohort references and drop all execution-error messages."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == 'error':
                kind = item.get('type') if isinstance(item, dict) else None
                if not isinstance(kind, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]{0,127}', kind):
                    kind = 'UnspecifiedError'
                result[key] = {'type': kind, 'message': _ERROR_TEXT, 'message_redacted': True}
                redactions[0] += 1
            else:
                result[key] = transform(item, old_cohort, new_cohort, redactions=redactions)
        return result
    if isinstance(value, list):
        return [transform(item, old_cohort, new_cohort, redactions=redactions) for item in value]
    if isinstance(value, str) and value == old_cohort:
        return new_cohort
    return value


def _without_error_text(value):
    if isinstance(value, dict):
        return {key: ({'type': item.get('type')} if key == 'error' and isinstance(item, dict)
                      else _without_error_text(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_without_error_text(item) for item in value]
    return value


def export(raw_table, runs, output):
    """Write a new directory, a validated public table and an export receipt."""
    raw_table = Path(raw_table).resolve(strict=True)
    directories = [Path(p).resolve(strict=True) for p in runs]
    output = Path(output).resolve()
    if (output.exists() or any(output.is_relative_to(p) for p in directories)
            or output == raw_table.parent):
        raise ValueError('A new output directory outside the raw repeat directories is required')
    source = collection._read_json(raw_table)
    collection.validate_table(source)
    before = {'table.json': sha(raw_table)}
    by_index = {}
    for directory in directories:
        index = collection._read_json(directory / 'plan.json')['repeat_index']
        if index in by_index:
            raise ValueError('Duplicate source repeat')
        by_index[index] = directory
        for name in _FILES:
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('Source artifacts must be regular files')
            before[f'repeat-{index}/{name}'] = sha(path)
    with tempfile.TemporaryDirectory(prefix='drivebench-public-export-') as temporary:
        temporary = Path(temporary)
        rebuilt = collection.merge(directories, temporary / 'raw-rebuilt.json')
        if {k: v for k, v in source.items() if k != 'created_at'} != {
                k: v for k, v in rebuilt.items() if k != 'created_at'}:
            raise ValueError('Raw table is not the table derived from the supplied repeat artifacts')
        runtime, gpu_ids = public_runtime(source['runtime'])
        old_cohort, new_cohort = source['cohort_id'], runtime['cohort_id']
        output.mkdir(parents=True, exist_ok=False)
        derived_directories, mappings, redaction_counts = [], [], {}
        for index, directory in sorted(by_index.items()):
            target = output / f'repeat-{index}'
            target.mkdir()
            derived_directories.append(target)
            for name in _FILES:
                count = [0]
                if name == 'runtime.json':
                    document = runtime
                elif name == 'trials.jsonl':
                    with (target / name).open('x') as stream:
                        for line in (directory / name).read_text().splitlines():
                            event = transform(json.loads(line), old_cohort, new_cohort, redactions=count)
                            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
                    document = None
                else:
                    document = transform(collection._read_json(directory / name), old_cohort,
                                         new_cohort, redactions=count)
                    if name == 'completed.json':
                        document['artifact_sha256'] = {
                            artifact: sha(target / artifact) for artifact in collection._ARTIFACTS}
                        document['public_export'] = {
                            'derived_from_completed_sha256': before[f'repeat-{index}/completed.json'],
                            'measurements_rerun': False,
                            'hardware_identifiers_pseudonymized': True,
                            'execution_error_messages_redacted': True}
                if document is not None:
                    write_json(target / name, document)
                relative = f'repeat-{index}/{name}'
                redaction_counts[relative] = count[0]
                mappings.append({'artifact': relative, 'source_sha256': before[relative],
                                 'derived_sha256': sha(target / name)})
        derived = collection.merge(derived_directories, temporary / 'derived-merged.json')
        # Reversing only the cohort remapping must leave every measurement field
        # identical, apart from the explicitly removed error text.
        reversed_rows = transform(derived['rows'], new_cohort, old_cohort, redactions=[0])
        if _without_error_text(reversed_rows) != _without_error_text(source['rows']):
            raise ValueError('An exported measurement field changed')
        derived['public_export'] = {
            'kind': 'sanitized_copy_of_recorded_measurements', 'source_table_sha256': before['table.json'],
            'source_cohort_id': old_cohort, 'source_table_created_at': source['created_at'],
            'measurements_rerun': False, 'measurement_values_unchanged': True,
            'hardware_identifiers_pseudonymized': True, 'execution_error_messages_redacted': True,
            'provenance_receipt': 'export-receipt.json'}
        collection.validate_table(derived)
        write_json(output / 'table.json', derived)
        mappings.append({'artifact': 'table.json', 'source_sha256': before['table.json'],
                         'derived_sha256': sha(output / 'table.json')})
        # Confirm no raw hardware identifier survived anywhere in public files.
        for path in output.rglob('*'):
            if path.is_file():
                text = path.read_text().lower()
                if any(raw_id in text or raw_id.replace('-', '') in text for raw_id in gpu_ids):
                    raise ValueError('Unredacted physical GPU identity found in derived artifact')
        after = {'table.json': sha(raw_table)}
        for index, directory in by_index.items():
            after.update({f'repeat-{index}/{name}': sha(directory / name) for name in _FILES})
        if before != after:
            raise ValueError('Raw evidence changed during export')
        receipt = {'schema_version': 1, 'kind': 'public_operator_data_export', 'completed': True,
                   'exported_at': utc_now(), 'exporter_sha256': sha(__file__),
                   'source_cohort_id': old_cohort, 'derived_cohort_id': new_cohort,
                   'gpu_identifier_policy': 'SHA256 of normalized physical UUID with drivebench-public-gpu-v1 prefix; same GPU maps consistently',
                   'pseudonymized_gpu_count': len(gpu_ids), 'raw_gpu_identifiers_in_receipt': False,
                   'error_policy': 'Error class retained; all execution error messages replaced and marked message_redacted',
                   'error_redactions_by_artifact': redaction_counts,
                   'measurements_rerun': False, 'measurement_values_unchanged': True,
                   'raw_artifact_hashes_unchanged': True, 'source_table_validated': True,
                   'derived_table_validated': True, 'artifacts': mappings}
        write_json(output / 'export-receipt.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', required=True, type=Path)
    parser.add_argument('--runs', required=True, type=Path, nargs='+')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = export(args.table, args.runs, args.output)
    print(json.dumps({'completed': True, 'measurements_rerun': False,
                      'table_sha256': sha(args.output / 'table.json'),
                      'receipt_sha256': sha(args.output / 'export-receipt.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
