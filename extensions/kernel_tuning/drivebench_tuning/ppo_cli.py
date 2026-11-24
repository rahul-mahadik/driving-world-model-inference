"""Prospectively freeze, train and evaluate the CPU PPO supplement."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .ppo import (SETTINGS, TRAINING_SEEDS, SB3_VERSION, PPOSelector, TrainingSnapshot,
                  digest, file_digest, train_seed)
from .replay import MeasuredTable
from .replay_policy import run_replay_policy
from .generalization_cli import add_budget_metrics
from .generalization_analysis import freeze_modal_selections
from .replay_cli import comparison_summary


def _write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write("\n")


def source_hashes():
    package = Path(__file__).parent
    return {name: file_digest(package / name) for name in (
        "ppo.py", "ppo_cli.py", "search.py", "replay.py", "replay_policy.py",
        "evaluation.py", "suite.py", "generalization_cli.py", "generalization_analysis.py")}


def make_protocol(table):
    snapshot = TrainingSnapshot(table)
    return {"schema_version": 1, "kind": "operator_ppo_preregistered_supplement",
            "source_hashes": source_hashes(), "table_sha256": table.sha256,
            "suite_sha256": table.suite.sha256, "source_cohort_id": table.cohort_id,
            "train_tasks": [t.task_id for t in snapshot.tasks],
            "test_tasks": [table.suite.tasks[i].task_id for i in table.suite.splits["test"]],
            "training_source_records_sha256": snapshot.provenance["source_records_sha256"],
            "training_row_count": snapshot.provenance["row_count"],
            "library": "stable-baselines3", "library_version": SB3_VERSION,
            "training_seeds": list(TRAINING_SEEDS), "settings": SETTINGS,
            "hyperparameter_selection": "none; fixed before training, no validation or test tuning",
            "evaluation": {"split": "test", "repeats": 20, "trials": 8, "seed": 2101,
                           "deterministic_actions": True, "untrained_seed": TRAINING_SEEDS[0],
                           "run_after_all_training_checkpoints_frozen": True,
                           "independent_gpu_measurements_added": 0},
            "live_policy": {"training_seed": TRAINING_SEEDS[0], "selection": "prospectively fixed",
                            "eight_trials_with_fresh_feedback": True, "fine_tuning": False},
            "limitations": ["five independent training seeds; twenty replay draws are not twenty trained policies",
                            "finite table reuse adds no GPU timing evidence",
                            "task holdout is by shape family, not operator family",
                            "checkpoint inference only sees public numeric observations"],
            "freeze_at": datetime.now(timezone.utc).isoformat()}


def validate_protocol(protocol, table):
    expected = make_protocol(table)
    if {k: v for k, v in protocol.items() if k != "freeze_at"} != {
            k: v for k, v in expected.items() if k != "freeze_at"}:
        raise ValueError("PPO protocol differs from fixed source, data or settings")
    if datetime.fromisoformat(protocol.get("freeze_at", "")).tzinfo is None:
        raise ValueError("PPO freeze requires a timezone-aware timestamp")


def run_training(table, protocol, output):
    validate_protocol(protocol, table)
    snapshot = TrainingSnapshot(table)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    _write(output / "protocol.json", protocol)
    _write(output / "train-only-provenance.json", snapshot.provenance)
    rows = []
    for seed in TRAINING_SEEDS:
        validate_protocol(protocol, table)
        print(json.dumps({"stage": "training", "seed": seed, "time": datetime.now(timezone.utc).isoformat()}), flush=True)
        manifest = train_seed(snapshot, output / f"seed-{seed}", seed,
                              initial_checkpoint=seed == TRAINING_SEEDS[0])
        rows.append({"seed": seed, "manifest_sha256": file_digest(output / f"seed-{seed}/manifest.json"),
                     "checkpoint_sha256": manifest["checkpoint_sha256"],
                     "parameters_changed": manifest["parameters_changed"],
                     "training_timesteps": manifest["training_timesteps"],
                     "wall_seconds": manifest["wall_seconds"]})
        print(json.dumps({"stage": "trained", **rows[-1]}), flush=True)
    completed = {"completed_at": datetime.now(timezone.utc).isoformat(), "all_seeds": rows,
                 "training_complete_before_test": True, "protocol_sha256": digest(protocol),
                 "live_seed": TRAINING_SEEDS[0], "test_outcomes_used_to_choose_seed": False}
    _write(output / "training-completed.json", completed)
    return completed


def run_evaluation(table, protocol, training, output):
    validate_protocol(protocol, table)
    training, output = Path(training), Path(output)
    completed = json.loads((training / "training-completed.json").read_text())
    if (completed.get("protocol_sha256") != digest(protocol)
            or [r["seed"] for r in completed.get("all_seeds", [])] != list(TRAINING_SEEDS)):
        raise ValueError("all prospectively declared training seeds must finish before TEST")
    for row in completed["all_seeds"]:
        sd = training / f"seed-{row['seed']}"
        if (file_digest(sd / "manifest.json") != row["manifest_sha256"]
                or file_digest(sd / "model.zip") != row["checkpoint_sha256"]):
            raise ValueError("training checkpoint/metadata changed before TEST")
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    descriptors = [(f"ppo-seed-{seed}", training / f"seed-{seed}/model.zip",
                    training / f"seed-{seed}/manifest.json") for seed in TRAINING_SEEDS]
    descriptors += [(f"ppo-untrained-seed-{TRAINING_SEEDS[0]}", training / f"seed-{TRAINING_SEEDS[0]}/untrained.zip",
                     training / f"seed-{TRAINING_SEEDS[0]}/untrained-manifest.json")]
    for name, checkpoint, manifest in descriptors:
        selector = PPOSelector.load(checkpoint, manifest, table.suite)
        report = add_budget_metrics(run_replay_policy(table, selector, name=name, split="test", repeats=20,
                                                       seed=2101, trials=8, memory_weight=0.0))
        _write(output / f"{name}.json", report)
        reports.append(report)
        print(json.dumps({"stage": "evaluated", "policy": name, "status": report["status"],
                          "episodes": len(report["episodes"])}), flush=True)
        if report["status"] != "completed":
            raise RuntimeError("PPO held-out evaluation incomplete; partial report retained")
    summary = summarize_reports(reports)
    _write(output / "summary.json", summary)
    _write(output / "frozen-modal-selections.json", freeze_modal_selections(table, reports))
    _write(output / "completed.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
                                       "policies": [r[0] for r in descriptors],
                                       "episodes": sum(len(r["episodes"]) for r in reports),
                                       "fresh_gpu_measurements": 0,
                                       "training_completed_receipt_sha256": file_digest(training / "training-completed.json")})
    return summary


def summarize_reports(reports):
    """Describe training provenance, rather than whether summarization trains."""
    summary = comparison_summary(reports)
    summary["policy_training_performed"] = any(r.get("training_performed") is True for r in reports)
    summary["rl_training_performed"] = any(r.get("rl_training_performed") is True for r in reports)
    summary["five_training_seeds_not_twenty"] = True
    summary["untrained_reference_not_used_for_selection"] = True
    summary["training_performed_during_evaluation"] = False
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "train", "evaluate"))
    p.add_argument("--table", required=True); p.add_argument("--protocol", required=True)
    p.add_argument("--output"); p.add_argument("--training")
    args = p.parse_args(argv)
    table = MeasuredTable.load(args.table)
    if args.command == "freeze":
        _write(args.protocol, make_protocol(table)); return 0
    protocol = json.loads(Path(args.protocol).read_text())
    if not args.output:
        p.error("--output required")
    if args.command == "train":
        run_training(table, protocol, args.output)
    else:
        if not args.training: p.error("--training required")
        run_evaluation(table, protocol, args.training, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
