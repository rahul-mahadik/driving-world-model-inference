#!/usr/bin/env python3
"""Validate a completed small Nsight Compute capture's wide raw CSV, CPU only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re


def analyze(directory):
    command=json.loads((directory/"command.json").read_text())
    argv=command["argv"]
    metrics=argv[argv.index("--metrics")+1].split(",")
    count=int(argv[argv.index("--launch-count")+1])
    text=(directory/"profiler.stdout.txt").read_text()
    header_index=text.find('"ID","Process ID"')
    if header_index < 0:
        raise ValueError("No wide raw CSV header")
    parsed=list(csv.DictReader(io.StringIO(text[header_index:])))
    units=next(row for row in parsed if not row.get("ID"))
    rows=[row for row in parsed if row.get("ID","").isdigit()]
    kernel_filter=argv[argv.index("--kernel-name")+1].removeprefix("regex:")
    nvtx_filter=argv[argv.index("--nvtx-include")+1]
    stages=nvtx_filter.split("/")
    nvtx_column=next(key for key in rows[0] if key.startswith("thread Domain:"))
    results=[]
    errors=[]
    if command.get("returncode") != 0:
        errors.append("Command did not return zero")
    if len(rows) != count:
        errors.append(f"Expected {count} kernels, found {len(rows)}")
    if not (directory/"kernels.ncu-rep").is_file() or not (directory/"kernels.ncu-rep").stat().st_size:
        errors.append("Missing or empty binary report")
    launch_keys=["launch__registers_per_thread","launch__registers_per_thread_allocated",
                 "launch__shared_mem_per_block_dynamic","launch__shared_mem_per_block_driver",
                 "launch__shared_mem_per_block_allocated","launch__occupancy_limit_registers",
                 "launch__occupancy_limit_shared_mem","launch__occupancy_limit_warps",
                 "launch__occupancy_limit_blocks","launch__sm_count","launch__waves_per_multiprocessor",
                 "sm__maximum_warps_per_active_cycle_pct","profiler__replayer_passes",
                 "profiler__replayer_bytes_mem_backed_up.sum"]
    for row in rows:
        values={}
        for metric in metrics:
            value=float(row[metric].replace(",",""))
            if not math.isfinite(value) or value <= 0:
                errors.append(f"Kernel {row['ID']} invalid/nonpositive metric {metric}")
            values[metric]={"value":value,"raw_value":row[metric],"unit":units.get(metric)}
        if not re.search(kernel_filter,row["Kernel Name"]):
            errors.append(f"Kernel {row['ID']} fails requested name filter")
        stack=row[nvtx_column]
        positions=[stack.find(stage) for stage in stages]
        if -1 in positions or positions != sorted(positions):
            errors.append(f"Kernel {row['ID']} lacks ordered NVTX stack")
        results.append({"id":int(row["ID"]),"name":row["Kernel Name"],"grid":row["Grid Size"],
                        "block":row["Block Size"],"device":row["device__attribute_display_name"],
                        "compute_capability":row["CC"],"nvtx_stack":stack,"metrics":values,
                        "additional_launch_and_replay_values":{key:row.get(key) for key in launch_keys}})
    source_files={name:{"bytes":(directory/name).stat().st_size,
                        "sha256":hashlib.file_digest((directory/name).open("rb"),"sha256").hexdigest()}
                  for name in ("command.json","profiler.stdout.txt","kernels.ncu-rep")}
    return {"profiled":True,"headline_eligible":False,"accepted_upstream_baseline":False,
            "scope":"initial adapter before baseline autocast-cache faithfulness correction",
            "command_returncode":command.get("returncode"),"started_at":command.get("started_at"),
            "finished_at":command.get("finished_at"),"requested_metrics":metrics,
            "requested_kernel_count":count,"observed_kernel_count":len(rows),"nvtx_filter":nvtx_filter,
            "hardware_counters_collected":not errors,"passed":not errors,"errors":errors,
            "source_files":source_files,"kernels":results,
            "method_notes":["Metrics describe the two selected replayed kernels, not model-wide utilization.",
                            "Raw CSV also contains related metrics and launch/device metadata beyond the seven requested names.",
                            "Replay and profiler controls affect durations; no unprofiled timing or speedup is inferred."]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.directory)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if not result["passed"]:
        raise SystemExit(1)
