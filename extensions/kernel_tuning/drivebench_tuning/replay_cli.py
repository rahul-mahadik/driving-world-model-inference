"""Compare fixed, grid and random policies on a measured operator table."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from .evaluation import grid_policy, random_policy
from .replay import MeasuredTable
from .replay_policy import fixed_policy, run_replay_policy
from .suite import load_suite


def selection_manifest(table, reports):
    """Deduplicate selected pairs for a separately run, fresh GPU confirmation.

    Mapping every completed episode to its chosen pair is retained. A reference
    selection needs only a fresh baseline. This manifest is a request, not proof
    of confirmation, and cannot establish transfer to another physical GPU.
    """
    pairs = {}
    for report in reports:
        if report["table_sha256"] != table.sha256 or report["suite_sha256"] != table.suite.sha256:
            raise ValueError("selection reports refer to another table or suite")
        for episode in report["episodes"]:
            if episode["status"] != "completed":
                continue
            info = episode["final_info"]
            key = (episode["task_id"], info["incumbent_candidate_id"])
            entry = pairs.setdefault(key, {"task_id": key[0], "candidate_id": key[1], "selected_by": []})
            entry["selected_by"].append({"policy": report["policy_name"],
                                        "episode_index": episode["episode_index"], "seed": episode["seed"]})
    tasks = {t.task_id: t.to_dict() for t in table.suite.tasks}
    candidates = {c.candidate_id: c.to_dict() for c in table.suite.candidates}
    rows = []
    for key, entry in sorted(pairs.items(), key=lambda pair: (pair[0][0], pair[0][1] or "")):
        rows.append({**entry, "task": tasks[key[0]],
                     "candidate": None if key[1] is None else candidates[key[1]]})
    return {"schema_version": 1, "kind": "selected_configuration_confirmation_request",
            "table_sha256": table.sha256, "suite_sha256": table.suite.sha256,
            "cohort_id": table.cohort_id, "confirmed": False, "selections": rows,
            "expected_input_digests": {task_id: table.row(task_id, None, table.repeat_indices[0])["input_digest"]
                                       for task_id in sorted({r["task_id"] for r in rows})},
            "measurement_requirement": "new process, inputs regenerated, correctness rerun, eager and selected candidate paired",
            "hardware_requirement": "same physical GPU/runtime; other hardware creates a separate transfer study"}


def comparison_summary(reports):
    policies = []
    for report in reports:
        curves = defaultdict(list)
        failure = []
        for episode in report["episodes"]:
            if episode["status"] != "completed":
                continue
            for row in episode["table_analysis"]["curve"]:
                curves[row["trials_used"]].append(row)
            failure.append(episode["failed_or_incorrect_trials"] + episode["rejected_policy_actions"])
        policies.append({"policy": report["policy_name"], "status": report["status"],
                         "completed_episodes": len(failure), "counts": report["counts"],
                         "mean_failed_or_rejected_turns": sum(failure) / len(failure) if failure else None,
                         "budget_curve": [{"trials_used": budget,
                                           "mean_regret": sum(r["table_regret"] for r in rows) / len(rows),
                                           "median_cost": median(r["table_selection_cost"] for r in rows),
                                           "mean_selection_failure_rate": sum(r["selection_failure_rate"] for r in rows) / len(rows)}
                                          for budget, rows in sorted(curves.items())]})
    return {"schema_version": 1, "backend": "measured_replay", "policies": policies,
            "fresh_gpu_measurements": 0, "policy_training_performed": False,
            "live_selected_confirmation_performed": False,
            "scope": "Descriptive search on one finite recorded table; policy seeds do not add GPU samples."}


def write_json(path, document):
    with Path(path).open("x") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--trials", type=int, default=4)
    args = parser.parse_args(argv)
    table = MeasuredTable.load(args.table, suite=load_suite(args.suite))
    if not 0 <= args.seed < 2**32 or not 1 <= args.repeats <= 10000 or not 1 <= args.trials < len(table.suite.candidates):
        raise ValueError("seed, repeats or trial budget outside admitted bounds")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", {"backend": "measured_replay", "table_sha256": table.sha256,
        "suite_sha256": table.suite.sha256, "split": args.split, "repeats": args.repeats,
        "seed": args.seed, "trials": args.trials, "policies": ["fixed-compiled", "grid", "random"],
        "fresh_gpu_measurements": 0, "policy_training_performed": False})
    reports = []
    for name, policy in (("fixed-compiled", fixed_policy()), ("grid", grid_policy), ("random", random_policy(args.seed))):
        report = run_replay_policy(table, policy, name=name, split=args.split, repeats=args.repeats,
                                   seed=args.seed, trials=args.trials)
        reports.append(report)
        write_json(args.output / (name + ".json"), report)
    write_json(args.output / "comparison.json", comparison_summary(reports))
    write_json(args.output / "confirmation-request.json", selection_manifest(table, reports))
    complete = all(r["status"] == "completed" for r in reports)
    write_json(args.output / ("completed.json" if complete else "failed.json"),
               {"completed": complete, "backend": "measured_replay", "fresh_gpu_measurements": 0})
    print(json.dumps({"completed": complete, "backend": "measured_replay", "output": str(args.output)}))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
