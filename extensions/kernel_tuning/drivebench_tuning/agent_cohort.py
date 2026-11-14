"""Selection-only submission and a trusted final-cohort admission receipt.

This is bookkeeping, not a scheduler, GPU launcher, sandbox or resource guard.
All sessions must be terminal before the caller can obtain a final batch plan.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime
import json
import math
from pathlib import Path
import time

from .agent_scoring import QualityPolicy
from .agent_session import AgentSession, SessionPolicy, _read, _write_new
from .vista_agent import digest, validate_settings


POLICIES = ('llm', 'random', 'grid')
ROLES = ('reference', 'llm_selected', 'random_selected', 'grid_selected')
COMPARISON_MODE = 'three_policy_comparison'
SINGLE_LLM_MODE = 'single_llm_feasibility'


def study_policies(study_mode=COMPARISON_MODE, policy_names=POLICIES):
    """Validate one of two explicit study contracts; missing sessions never opt in."""
    if type(study_mode) is not str or type(policy_names) not in (list, tuple):
        raise ValueError('explicit study mode and ordered policy names required')
    if any(type(name) is not str for name in policy_names):
        raise ValueError('policy names must be strings')
    expected = {COMPARISON_MODE: POLICIES, SINGLE_LLM_MODE: ('llm',)}.get(study_mode)
    if expected is None or tuple(policy_names) != expected:
        raise ValueError('study mode and policy names differ')
    return expected


def final_plan_policies(plan):
    """Legacy plans mean all three policies; explicit contracts are hash-bound."""
    declared = {'study_mode', 'policy_names'} & set(plan)
    if declared and declared != {'study_mode', 'policy_names'}:
        raise ValueError('final plan must declare both study mode and policy names')
    names = study_policies(plan['study_mode'], plan['policy_names']) if declared else POLICIES
    if set(plan['selections']) != set(names):
        raise ValueError('final selections differ from declared study policies')
    roles = {'reference', *(name + '_selected' for name in names)}
    if any(entry['role'] not in roles for entry in plan['batches']):
        raise ValueError('final batch role outside declared study policies')
    return names


def selection_only_executor(trial_executor):
    """Wrap a trusted provisional trial; submit saves selection without inference.

    Session completion here means selection was recorded. The separate final
    admission check also requires an eligible measured trial for that proposal.
    """
    if not callable(trial_executor):
        raise ValueError('trusted trial executor required')
    def execute(kind, proposal, context):
        if kind == 'trial':
            return trial_executor(proposal, context)
        if kind != 'final':
            raise ValueError('this cohort budgets no live profile executions')
        return {'cohort_id': context['cohort_id'], 'status': 'ok',
                'selection_only': True, 'evaluation_performed': False, 'quality_assessed': False,
                'proposal_id': proposal['proposal_id'],
                'candidate_sha256': digest(proposal['config'])}
    return execute


def _sha(value, label):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{label} must be a SHA256')
    return value


def freeze_final_plan(path, *, sessions, quality_policy, block_orders,
                      source_manifest_sha256, inputs_manifest_sha256, deadline_unix, synthetic=False,
                      clock=time.time, study_mode=COMPARISON_MODE, policy_names=POLICIES):
    """Validate every declared closed selection and write an exclusive final plan.

    sessions maps each exact policy name to (directory, externally pinned
    SessionPolicy). Trial receipts must bind candidate/identity/quality hashes,
    status='ok', and screening_status='no_observed_breach'. That is provisional
    subset screening, never a full quality pass. No agent chooses final inputs.
    The caller must protect this plan and enforce its deadline and batch order.
    The default retains the three-policy contract. Single-LLM feasibility must
    explicitly declare its mode and names in both this call and the externally
    pinned session metadata, before that session begins selecting a candidate.
    """
    names = study_policies(study_mode, policy_names)
    roles = ('reference', *(name + '_selected' for name in names))
    if set(sessions) != set(names) or not isinstance(quality_policy, QualityPolicy):
        raise ValueError('all three pinned sessions required by default; explicit study sessions and frozen quality policy must match')
    quality_policy = QualityPolicy.from_dict(quality_policy.to_dict())
    quality = quality_policy.to_dict()
    if quality['calibration']['status'] != 'ready':
        raise ValueError('quality calibration is inconclusive')
    _sha(source_manifest_sha256, 'source manifest')
    _sha(inputs_manifest_sha256, 'inputs manifest')
    at = clock()
    if (type(at) not in (int, float) or not math.isfinite(at) or
        type(deadline_unix) not in (int, float) or not math.isfinite(deadline_unix) or at >= deadline_unix):
        raise ValueError('positive immutable time remaining required')
    last_clock = at
    def checked_clock():
        nonlocal last_clock
        value = clock()
        if type(value) not in (int, float) or not math.isfinite(value) or value < last_clock:
            raise ValueError('cohort clock must be monotone')
        last_clock = value
        return value
    frozen = datetime.fromisoformat(quality['plan']['frozen_at']).timestamp()
    partitions = quality['plan']['timing']['block_scenes']
    if partitions is None:
        raise ValueError('explicit final scene partitions required')
    if set(block_orders) != set(partitions):
        raise ValueError('one declared role order for every timing block required')
    for order in block_orders.values():
        if type(order) is not list or len(order) != len(roles) or set(order) != set(roles):
            raise ValueError('each block must order every role exactly once')
    if quality['plan']['replicates'] != [0, 1]:
        raise ValueError('balanced cohort adapter requires repeats 0 and 1')
    selections = {}
    cohorts = set()
    reference = None
    shared_search_contract = None
    # Hold every session lock through the write, so the snapshots cannot race
    # another compliant controller. Host filesystem access remains trusted.
    with ExitStack() as stack:
        for name in names:
            directory, expected_policy = sessions[name]
            if not isinstance(expected_policy, SessionPolicy):
                raise ValueError('externally pinned SessionPolicy required')
            session = stack.enter_context(AgentSession.open(directory,
                executor=lambda *args: (_ for _ in ()).throw(RuntimeError('read-only admission')),
                expected_policy=expected_policy, clock=checked_clock))
            state = session.state
            if state['status'] != 'completed' or state['pending'] is not None:
                raise ValueError('all policy selections must complete before final evaluation')
            manifest = _read(Path(directory)/'policy.json')
            if not frozen <= manifest['created_at'] <= at:
                raise ValueError('selection must start after quality freeze and before admission')
            metadata = expected_policy.to_dict()['metadata']
            if study_mode == SINGLE_LLM_MODE and (
                metadata.get('study_mode') != study_mode or
                metadata.get('policy_names') != list(names)):
                raise ValueError('single-LLM study must be pinned in session metadata before selection')
            if (metadata.get('quality_policy_sha256') != quality_policy.sha256 or
                metadata.get('identity_sha256') != quality_policy.identity_sha256 or
                metadata.get('source_manifest_sha256') != source_manifest_sha256 or
                metadata.get('inputs_manifest_sha256') != inputs_manifest_sha256):
                raise ValueError('selection policy identity drift')
            if expected_policy.max_profiles != 0:
                raise ValueError('unbudgeted profiling is not allowed in this cohort')
            config = validate_settings(expected_policy.to_dict()['reference_config'], synthetic=synthetic)
            if reference is not None and reference != config:
                raise ValueError('policies must share one reference config')
            reference = config
            contract = (expected_policy.to_dict()['allowed_config_values'], expected_policy.max_evaluations,
                        expected_policy.final_evaluations, expected_policy.max_tool_calls,
                        expected_policy.wall_seconds, expected_policy.final_reserve_seconds)
            if shared_search_contract is not None and contract != shared_search_contract:
                raise ValueError('policies must share candidate choices and trial budgets')
            shared_search_contract = contract
            cohorts.add(expected_policy.cohort_id)
            selected = state['selected_proposal_id']
            proposal = _read(Path(directory)/'proposals'/f'{selected}.json')
            chosen = validate_settings(proposal['config'], synthetic=synthetic)
            events = [_read(p) for p in sorted((Path(directory)/'events').glob('*.json'))]
            pending = None
            eligible = []
            final_receipt = None
            for event in events:
                if event['type'] == 'execution_started':
                    pending = event['payload']
                elif event['type'] in ('tool_finished', 'tool_failed'):
                    result = event['payload'].get('result', {})
                    if pending and event['type'] == 'tool_finished':
                        if (pending['kind'] == 'trial' and pending['proposal_id'] == selected and
                            result.get('status') == 'ok' and
                            result.get('screening_status') == 'no_observed_breach' and
                            result.get('candidate_sha256') == digest(chosen) and
                            result.get('quality_policy_sha256') == quality_policy.sha256 and
                            result.get('identity_sha256') == quality_policy.identity_sha256):
                            eligible.append({'event_sequence':event['sequence'],'receipt_sha256':digest(result)})
                        if pending['kind'] == 'final':
                            final_receipt = result
                    pending = None
            if not eligible:
                raise ValueError('selected proposal has no eligible measured provisional trial')
            expected = {'cohort_id':expected_policy.cohort_id,'status':'ok','selection_only':True,
                        'evaluation_performed':False,'quality_assessed':False,'proposal_id':selected,'candidate_sha256':digest(chosen)}
            if final_receipt != expected:
                raise ValueError('submission must only lock selection, without final evaluation')
            selections[name] = {'proposal_id':selected,'config':chosen,'candidate_sha256':digest(chosen),
                'session_policy_sha256':state['policy_sha256'],'session_ledger_sha256':state['ledger_sha256'],
                'selected_at':events[-1]['at'],'eligible_trials':eligible}
        if len(cohorts) != 1:
            raise ValueError('cross-cohort selection is forbidden')
        end = checked_clock()
        if not at <= end < deadline_unix:
            raise ValueError('final admission deadline or clock monotonicity failed')
        batches = []
        for block in sorted(partitions):
            for rep in [0, 1]:
                order = block_orders[block] if rep == 0 else list(reversed(block_orders[block]))
                for role in order:
                    config = reference if role == 'reference' else selections[role.removesuffix('_selected')]['config']
                    batches.append({'batch_index':len(batches),'block_id':block,'replicate':rep,'role':role,
                        'scenes':partitions[block],'seeds':quality['plan']['seeds'],'config':config,
                        'candidate_sha256':digest(config)})
        body = {'schema_version':1,'cohort_id':next(iter(cohorts)),'quality_policy_sha256':quality_policy.sha256,
            'identity_sha256':quality_policy.identity_sha256,'source_manifest_sha256':source_manifest_sha256,
            'inputs_manifest_sha256':inputs_manifest_sha256,
            'all_selections_locked_at':end,'deadline_unix':deadline_unix,'selections':selections,'batches':batches,
            'shared_reference_comparisons_independent':False,'quality_preserving_claim':False,
            'gpu_execution_performed':False,'synthetic':synthetic}
        if study_mode == SINGLE_LLM_MODE:
            body.update(study_mode=study_mode, policy_names=list(names))
        receipt = {**body,'sha256':digest(body)}
        _write_new(Path(path),receipt)
    return receipt
