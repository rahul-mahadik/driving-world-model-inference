#!/usr/bin/env python3
"""Verify installed CPU demos outside the source tree; no GPU or model API calls.

Run with the Python interpreter of the environment being checked. This script
does not install packages, download assets, provision resources or certify a GPU
study. Each existing demo owns its workflow and retains its own failure evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def write(path: Path, document: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run(output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a new output directory; prior attempts are preserved")
    root = output.resolve()
    repository = Path(__file__).resolve().parents[2]
    if root == repository or repository in root.parents:
        raise ValueError("Use an output directory outside this source checkout")
    installed = {}
    for distribution_name in ("drivebench", "drivebench-kernel-tuning"):
        distribution = metadata.distribution(distribution_name)
        direct_url = distribution.read_text("direct_url.json")
        if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
            raise ValueError("Use wheel installations, not editable installs")
        installed[distribution_name] = distribution.version
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (root / "home").mkdir()
    # Explicit minimal environment: provider/API secrets, PYTHONPATH and user
    # caches are not inherited by the demo subprocesses. This is not an OS sandbox.
    environment = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP")
                   if key in os.environ}
    environment.update(HOME=str(root / "home"), USERPROFILE=str(root / "home"),
                       CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", PYTHONHASHSEED="0")
    plan = {
        "schema_version": 1,
        "kind": "installed_cpu_reproduction_check",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": True,
        "gpu_validation": False,
        "real_model_or_llm_execution": False,
        "python_version": platform.python_version(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "packages": installed,
        "optional_packages_present": {name: util.find_spec(name) is not None for name in ("torch", "triton", "openai", "lpips")},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "commands": [
            {"entrypoint": "drivebench-demo", "arguments": ["--output", "video"]},
            {"entrypoint": "drivebench-search", "arguments": ["run", "--backend", "synthetic", "--split", "validation", "--policy", "random", "--output", "search"]},
            {"entrypoint": "drivebench-agent-demo", "arguments": ["--output", "agent"]},
        ],
        "scope": "Installed-package CPU integration only. Synthetic timings/rewards are not scientific measurements.",
    }
    write(root / "plan.json", plan)
    completed = []
    start = time.monotonic()
    try:
        for index, item in enumerate(plan["commands"]):
            entrypoint = item["entrypoint"]
            # -I prevents cwd/user-site/PYTHONPATH from substituting checkout code.
            loader = ("from importlib.metadata import entry_points; import sys; "
                      "name=sys.argv.pop(1); "
                      "matches=list(entry_points(group='console_scripts',name=name)); "
                      "assert len(matches)==1, 'Expected one installed entry point'; "
                      "sys.exit(matches[0].load()())")
            log = root / "logs" / f"{index + 1:02d}-{entrypoint}.log"
            step_start = time.monotonic()
            with log.open("x", encoding="utf-8") as stream:
                subprocess.run([sys.executable, "-I", "-c", loader, entrypoint, *item["arguments"]],
                               cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                               check=True, timeout=360)
            completed.append({"entrypoint": entrypoint, "returncode": 0,
                              "wall_seconds": time.monotonic() - step_start,
                              "log": log.relative_to(root).as_posix()})
        video = json.loads((root / "video/demo.json").read_text())
        search = json.loads((root / "search/completed.json").read_text())
        agent = json.loads((root / "agent/demo-report.json").read_text())
        if not (video["status"] == "completed" and video["synthetic"] is True
                and video["gpu_measurement"] is False and video["full_model_inference"] is False
                and search["completed"] is True and search["synthetic"] is True
                and search["backend"] == "synthetic"
                and agent["final_state"]["status"] == "completed"
                and agent["real_gpu_validation"] is False and agent["llm_demonstration"] is False
                and agent["real_quality_calibrated"] is False):
            raise RuntimeError("Demo receipt is incomplete or exceeds the synthetic claim boundary")
        receipts = ["video/demo.json", "search/completed.json", "agent/demo-report.json"]
        result = {**plan, "status": "passed", "completed_at": datetime.now(timezone.utc).isoformat(),
                  "wall_seconds": time.monotonic() - start, "completed_commands": completed,
                  "receipts_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in receipts}}
        write(root / "installation-check.json", result)
        return result
    except Exception as error:
        write(root / "failed.json", {**plan, "status": "failed", "error_type": type(error).__name__,
                                      "completed_commands": completed, "wall_seconds": time.monotonic() - start})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="A new directory outside the source checkout")
    arguments = parser.parse_args()
    result = run(arguments.output)
    print(json.dumps({"status": result["status"], "synthetic": True,
                      "gpu_validation": False, "commands_passed": len(result["completed_commands"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
