"""Run the complete synthetic CPU workflow from an installed package.

This is a functional demonstration, not a model, GPU, or driving-quality result.
Every existing entrypoint runs in its own subprocess with absolute input paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from drivebench.config import merge


DEMO_PROTOCOL = "synthetic-cpu-demo-v1"
ASSETS = ("data.yaml", "baseline.yaml", "candidate.yaml", "tolerances.yaml")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run_demo(output: str | Path) -> dict:
    """Create a new output root and run prepare, two benchmarks, evaluate, report.

    Existing roots, including empty directories and symlinks, are refused. Failed
    steps keep their logs and partial artifacts; choose another root to retry.
    """
    requested = Path(output).expanduser()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"Demo output already exists; choose a new path: {requested}")
    root = requested.resolve()
    root.mkdir(parents=True, exist_ok=False)
    configs, data, runs, logs = (root / name for name in ("configs", "data", "runs", "logs"))
    configs.mkdir()
    logs.mkdir()
    asset_root = resources.files("drivebench_demo").joinpath("assets")
    asset_bytes = {name: asset_root.joinpath(name).read_bytes() for name in ASSETS}
    documents = {name: yaml.safe_load(content) for name, content in asset_bytes.items()}
    data_config = documents["data.yaml"]
    baseline = documents["baseline.yaml"]
    baseline["data"] = {"manifest": str(data / "smoke.jsonl"),
                        "root": str(data / "synthetic_frames"), "verify_checksums": True}
    candidate = merge(baseline, documents["candidate.yaml"])
    for name, document in (("data.yaml", data_config), ("baseline.yaml", baseline),
                           ("candidate.yaml", candidate), ("tolerances.yaml", documents["tolerances.yaml"])):
        with (configs / name).open("x", encoding="utf-8") as stream:
            yaml.safe_dump(document, stream, sort_keys=False)

    paths = {"root": str(root), "data": str(data), "baseline": str(runs / "baseline"),
             "candidate": str(runs / "candidate"), "evaluation": str(runs / "evaluation.json"),
             "report": str(runs / "report.html"), "report_data": str(runs / "report.data.json"),
             "logs": str(logs)}
    plan = {"schema_version": 1, "protocol": DEMO_PROTOCOL, "created_at": _now(),
            "synthetic": True, "gpu_measurement": False, "full_model_inference": False,
            "quality_preserving": False, "paths": paths,
            "note": "Artificial CPU fixtures exercise software contracts only; no model or GPU performance claims.",
            "asset_sha256": {name: hashlib.sha256(content).hexdigest()
                             for name, content in asset_bytes.items()}}
    _write_json(root / "plan.json", plan)
    commands = [
        ("prepare", "drivebench.prepare", ["--config", str(configs / "data.yaml"), "--synthetic", "--output", str(data)]),
        ("baseline", "drivebench.benchmark", ["--config", str(configs / "baseline.yaml"), "--output", paths["baseline"]]),
        ("candidate", "drivebench.benchmark", ["--config", str(configs / "candidate.yaml"), "--output", paths["candidate"]]),
        ("evaluate", "drivebench.evaluate", ["--runs", str(runs), "--reference", paths["baseline"],
                                               "--tolerances", str(configs / "tolerances.yaml"), "--review-limit", "4"]),
        ("report", "drivebench.report", ["--runs", str(runs)]),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed_steps = []
    for index, (name, module, arguments) in enumerate(commands, 1):
        command = [sys.executable, "-m", module, *arguments]
        log = logs / f"{index:02d}-{name}.log"
        try:
            with log.open("x", encoding="utf-8") as stream:
                subprocess.run(command, cwd=root, env=environment, stdout=stream,
                               stderr=subprocess.STDOUT, check=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as error:
            _write_json(root / "failed.json", {**plan, "status": "failed", "failed_at": _now(),
                        "step": name, "error_type": type(error).__name__, "error": str(error),
                        "log": str(log), "completed_steps": completed_steps})
            raise RuntimeError(f"Synthetic demo failed during {name}; retained log: {log}") from error
        completed_steps.append({"step": name, "log": str(log), "returncode": 0})
    result = {**plan, "status": "completed", "completed_at": _now(), "completed_steps": completed_steps}
    _write_json(root / "demo.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New directory for the synthetic CPU demonstration")
    args = parser.parse_args(argv)
    try:
        result = run_demo(args.output)
    except (FileExistsError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
