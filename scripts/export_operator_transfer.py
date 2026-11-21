#!/usr/bin/env python3
"""Derive a public transfer report without changing or adding measurements."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent / 'extensions/kernel_tuning'))
from drivebench_tuning.benchmark import utc_now
from drivebench_tuning.generalization import digest
from drivebench_tuning.generalization_analysis import transfer_analysis
from drivebench_tuning.replay import MeasuredTable

spec = importlib.util.spec_from_file_location('transfer_related_export', SCRIPTS/'export_related_operator_results.py')
related = importlib.util.module_from_spec(spec)
spec.loader.exec_module(related)
public = related.public


def export_transfer(source_table, target_table, public_source_table, public_target_table,
                    selections, raw_analysis, related_export, output):
    paths = [Path(p).resolve(strict=True) for p in (source_table, target_table, public_source_table,
                                                  public_target_table, selections, raw_analysis)]
    source_path, target_path, public_source_path, public_target_path, selection_path, analysis_path = paths
    directory, output = Path(related_export).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise ValueError('A new transfer publication directory is required')
    receipt_path = directory/'export-receipt.json'
    receipt = related.plots.read_json(receipt_path)[0]
    raw_source, raw_target, public_source, public_target = [MeasuredTable.load(p) for p in paths[:4]]
    identities = (related._verify_table_derivation(raw_source, public_source) |
                  related._verify_table_derivation(raw_target, public_target))
    if (receipt.get('source_table_sha256') != raw_source.sha256 or
            receipt.get('derived_table_sha256') != public_source.sha256 or
            receipt.get('modal_decisions_reconstructed_from_source_and_derived_reports') is not True):
        raise ValueError('Related export does not bind the source table and frozen modal decisions')
    artifact = receipt.get('frozen_modal_selection_artifact')
    if not isinstance(artifact, str) or Path(artifact).is_absolute() or '..' in Path(artifact).parts:
        raise ValueError('Safe frozen selection artifact required')
    derived_selection_path = directory/artifact
    bindings = [r for r in receipt['selection_bindings'] if r['artifact'] == artifact]
    if (len(bindings) != 1 or public.sha(selection_path) != bindings[0]['source_selection_sha256_as_recorded']
            or public.sha(derived_selection_path) != bindings[0]['derived_selection_sha256']):
        raise ValueError('Raw/public frozen selection bytes differ from export receipt')
    original = related.plots.read_json(selection_path)[0]
    derived = related.plots.read_json(derived_selection_path)[0]
    mapping = {raw_source.sha256: public_source.sha256, raw_source.cohort_id: public_source.cohort_id}
    public_reports = []
    for row in receipt['artifacts']:
        if 'source_canonical_sha256' not in row:
            continue
        relative = Path(row['artifact'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Safe policy artifact required')
        path = directory/relative
        report = related.plots.read_json(path)[0]
        if public.sha(path) != row['derived_sha256'] or digest(report) != row['derived_canonical_sha256']:
            raise ValueError('Derived policy hash differs from recorded canonical binding')
        mapping[row['source_canonical_sha256']] = row['derived_canonical_sha256']
        public_reports.append(report)
        paths.append(path)
    if related.rebind(original, mapping, {'error_messages': 0, 'provider_response_ids': 0}) != derived:
        raise ValueError('Public frozen decisions differ from raw decisions beyond publication bindings')
    related.validate_modal(derived, public_source, public_reports)
    paths.extend([receipt_path, derived_selection_path])
    before = {p: public.sha(p) for p in paths}
    recorded = related.plots.read_json(analysis_path)[0]
    expected_raw = transfer_analysis(raw_source, raw_target, original)
    if recorded != expected_raw:
        raise ValueError('Raw transfer analysis differs from its source tables or frozen decisions')
    result = transfer_analysis(public_source, public_target, derived)
    mapping.update({raw_target.sha256: public_target.sha256, digest(original): digest(derived)})
    if related.rebind(recorded, mapping, {'error_messages': 0, 'provider_response_ids': 0}) != result:
        raise ValueError('Public transfer analysis changed beyond publication bindings')
    related.unchanged(recorded, result)
    output.mkdir(parents=True, exist_ok=False)
    public.write_json(output/'analysis.json', result)
    record = {'schema_version': 1, 'kind': 'public_operator_transfer_export', 'completed': True,
        'exported_at': utc_now(), 'exporter_sha256': public.sha(__file__),
        'new_gpu_measurements': 0, 'training_rerun': False,
        'selection_updated': False, 'numeric_fields_and_trial_ids_unchanged': True,
        'raw_source_table_sha256': raw_source.sha256, 'public_source_table_sha256': public_source.sha256,
        'raw_target_table_sha256': raw_target.sha256, 'public_target_table_sha256': public_target.sha256,
        'raw_selection_file_sha256': public.sha(selection_path),
        'public_selection_file_sha256': public.sha(derived_selection_path),
        'raw_selection_canonical_sha256': digest(original), 'public_selection_canonical_sha256': digest(derived),
        'related_export_receipt_sha256': public.sha(receipt_path),
        'raw_analysis_sha256': public.sha(analysis_path), 'public_analysis_sha256': public.sha(output/'analysis.json'),
        'scope': 'Offline publication rebinding and numerical verification; the original GPU evidence is unchanged.'}
    for path in output.rglob('*'):
        if path.is_file() and any(identity in path.read_text().lower() for identity in identities):
            raise ValueError('Raw GPU identity remains in transfer publication')
    if any(public.sha(p) != sha for p, sha in before.items()):
        raise ValueError('An input artifact changed during transfer export')
    record['source_artifact_hashes_unchanged'] = True
    public.write_json(output/'export-receipt.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-table', 'target-table', 'public-source-table', 'public-target-table',
                 'selections', 'raw-analysis', 'related-export', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    receipt = export_transfer(**vars(parser.parse_args(argv)))
    print(__import__('json').dumps({'completed': receipt['completed'], 'new_gpu_measurements': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
