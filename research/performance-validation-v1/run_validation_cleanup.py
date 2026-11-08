"""Run the frozen validator with compiler cleanup between completed engines.

This changes validation process lifetime, not generation or comparisons. It never
resets caches inside an engine's normal/reverse/action-change executions. Every
reset and its memory measurements remain recorded, including on failed attempts.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys

from drivebench import validate
from drivebench.benchmarks.environment import source_identity

# Keep the frozen source identity explicit; no inference package is edited.
SOURCE = "d8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb"
PHASES = ("upstream", "refactored", "isolation", "uncached", "cached")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--memory-audit", required=True)
    args, remaining = parser.parse_known_args()
    if remaining.count("--mode") != 1 or remaining[remaining.index("--mode") + 1] != "all":
        raise ValueError("This driver requires the complete existing all-mode validator")
    if remaining.count("--output") != 1:
        raise ValueError("Exactly one fresh validation report is required")
    report_path = Path(remaining[remaining.index("--output") + 1])
    audit_path = Path(args.memory_audit)
    artifact_dir = report_path.parent / (report_path.stem + "-outputs")
    if report_path.exists() or artifact_dir.exists() or audit_path.exists() or source_identity() != SOURCE:
        raise ValueError("Fresh outputs and unchanged frozen inference source required")
    import torch
    import torch._dynamo as dynamo
    import torch._dynamo.utils as utils
    import torch._inductor.metrics as metrics

    original_cleanup = validate._cleanup_gpu
    completed = []
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("x") as audit:
        def record(value):
            audit.write(json.dumps(value, sort_keys=True) + "\n")
            audit.flush()

        def memory():
            torch.cuda.synchronize()
            return {"allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved()}

        def cleanup():
            phase = PHASES[len(completed)]
            before = memory()
            record({"event": "cleanup_started", "completed_phase": phase,
                    "at": datetime.now(timezone.utc).isoformat(), "before": before})
            counters = copy.deepcopy(dict(utils.counters))
            kernels = metrics.generated_kernel_count
            dynamo.reset()
            # Preserve evidence of compilation and fallback across all engines.
            if dict(utils.counters) != counters or metrics.generated_kernel_count != kernels:
                raise RuntimeError("Compiler reset changed execution evidence counters")
            original_cleanup()
            entry = {"event": "between_engine_cleanup", "completed_phase": phase,
                     "at": datetime.now(timezone.utc).isoformat(),
                     "before": before, "after": memory(),
                     "execution_counters_preserved": True}
            completed.append(entry)
            record(entry)

        validate._cleanup_gpu = cleanup
        record({"event": "start", "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_identity": SOURCE, "validator_arguments": remaining,
                "torch_version": torch.__version__,
                "dynamo_reset_source_sha256": hashlib.sha256(inspect.getsource(dynamo.reset).encode()).hexdigest(),
                "dynamo_module_sha256": hashlib.sha256(Path(dynamo.__file__).read_bytes()).hexdigest(),
                "scope": "Compiler reset only after an entire engine invocation returns; no numerical or inference changes"})
        try:
            code = validate.main(remaining)
            if len(completed) != len(PHASES) or not report_path.is_file():
                raise RuntimeError("All five validation engines must finish")
            record({"event": "validator_returned", "exit_code": code})
        except BaseException as error:
            record({"event": "failed", "error_type": type(error).__name__, "error": str(error),
                    "completed_phases": [x["completed_phase"] for x in completed]})
            raise
        finally:
            validate._cleanup_gpu = original_cleanup
    report = json.loads(report_path.read_text())
    if "validation_driver" in report:
        raise ValueError("Unexpected pre-existing validation driver provenance")
    report["validation_driver"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": "torch._dynamo.reset then frozen gc.collect/empty_cache between completed engines only",
        "memory_audit": str(audit_path.resolve()),
        "memory_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "completed_cleanup_phases": list(PHASES),
        "frozen_validator_source_unchanged": source_identity() == SOURCE,
        "within_engine_isolation_and_cache_checks_unchanged": True,
        "timing_evidence": False,
    }
    if not report["validation_driver"]["frozen_validator_source_unchanged"]:
        raise RuntimeError("Frozen inference source changed")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
