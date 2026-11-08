#!/usr/bin/env python3
"""CPU-only, read-only analysis of an exported Nsight Systems SQLite trace.

Kernel family labels are name-based heuristics, not hardware-counter claims.
All durations retain nanoseconds in the JSON. Interval unions avoid summing
concurrent activity as elapsed time. NVTX attribution uses the launch API's
host timestamp/thread, not GPU execution overlap with an asynchronous range.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(end - start for start, end in merged(intervals))


def family(name):
    lowered = name.lower()
    if "fmha" in lowered or "flash" in lowered:
        return "attention"
    if "direct_copy_kernel_cuda" in name or "CatArrayBatchedCopy" in name:
        return "copy_cast_concat"
    if "nchwToNhwc" in name or "nhwcToNchw" in name:
        return "layout_conversion"
    if any(term in name for term in ("RowwiseMoments", "layer_norm", "LayerNorm", "GroupNorm", "ComputeFusedParams")):
        return "normalization"
    if "cudnn" in lowered or "convol" in lowered:
        return "convolution"
    if "gemm" in lowered or "cutlass::Kernel" in name:
        return "matrix_multiply"
    if "elementwise" in lowered:
        return "other_elementwise"
    return "other"


def analyze(directory):
    path = directory / "timeline.sqlite"
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    nvtx = [dict(start=start, end=end, name=text or strings.get(text_id), tid=tid)
            for start, end, text, text_id, tid in db.execute(
                "SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL")]
    measured = [item for item in nvtx if item["name"] == "drivebench_measured_request"]
    if len(measured) != 1:
        raise ValueError("Expected exactly one measured request NVTX range")
    bounds = measured[0]
    stages = [item for item in nvtx if item["name"].startswith("drivebench::")
              and item["name"] != "drivebench::gpu_inference"]
    runtime = {corr: dict(start=start, end=end, tid=tid, name=strings[name_id])
               for start, end, corr, tid, name_id in db.execute(
                   "SELECT start,end,correlationId,globalTid,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    events = []
    for start, end, corr, name_id, stream in db.execute(
            "SELECT start,end,correlationId,demangledName,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        events.append(dict(start=start, end=end, correlation=corr, name=strings[name_id],
                           kind="kernel", stream=stream))
    transfer_names = {row[0]: row[2] for row in db.execute("SELECT * FROM ENUM_CUDA_MEMCPY_OPER")}
    for start, end, corr, size, kind, stream in db.execute(
            "SELECT start,end,correlationId,bytes,copyKind,streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
        events.append(dict(start=start, end=end, correlation=corr, name=transfer_names[kind],
                           kind="memcpy", bytes=size, stream=stream))
    for start, end, corr, size, stream in db.execute(
            "SELECT start,end,correlationId,bytes,streamId FROM CUPTI_ACTIVITY_KIND_MEMSET"):
        events.append(dict(start=start, end=end, correlation=corr, name="memset",
                           kind="memset", bytes=size, stream=stream))
    kernels = [event for event in events if event["kind"] == "kernel"]
    if any(event["start"] < bounds["start"] or event["end"] > bounds["end"] for event in events):
        raise ValueError("CUDA activity extends outside captured measured range")

    def summary(group):
        spans = [(event["start"], event["end"]) for event in group]
        return {"count": len(group), "sum_duration_ns": sum(end-start for start,end in spans),
                "union_duration_ns": duration(spans),
                "first_start_ns": min((start for start,_ in spans), default=None),
                "last_end_ns": max((end for _,end in spans), default=None)}

    families = defaultdict(list)
    by_name = defaultdict(list)
    for event in kernels:
        families[family(event["name"])].append(event)
        by_name[event["name"]].append(event)
    stage_events = defaultdict(list)
    for event in events:
        launch = runtime.get(event["correlation"])
        owners = [] if launch is None else [stage for stage in stages
            if stage["tid"] == launch["tid"] and stage["start"] <= launch["start"] < stage["end"]]
        owner = min(owners, key=lambda stage: stage["end"]-stage["start"])["name"] if owners else "unattributed"
        stage_events[owner].append(event)

    activity_union = merged((event["start"],event["end"]) for event in events)
    gaps = [(left[1],right[0]) for left,right in zip(activity_union,activity_union[1:])]
    initial_gap = (bounds["start"],activity_union[0][0])
    trailing_gap = (activity_union[-1][1],bounds["end"])
    first_by_start = defaultdict(list)
    for event in events:
        first_by_start[event["start"]].append(event)
    gap_classes = Counter()
    late_launch_prefix = 0
    for start,end in gaps:
        upcoming = first_by_start[end]
        api = [runtime.get(event["correlation"]) for event in upcoming]
        api = [launch for launch in api if launch is not None]
        if not api:
            kind = "missing_runtime_correlation"
        elif any(launch["end"] <= start for launch in api):
            kind = "next_activity_submitted_before_gap"
        elif any(launch["start"] <= start for launch in api):
            kind = "next_activity_api_in_progress_at_gap_start"
        else:
            kind = "next_activity_api_started_during_gap"
            late_launch_prefix += min(end,min(launch["start"] for launch in api)) - start
        gap_classes[kind] += end-start

    transfers = defaultdict(list)
    for event in events:
        if event["kind"] == "memcpy":
            transfers[event["name"]].append(event)
    api_groups = defaultdict(list)
    for launch in runtime.values():
        api_groups[launch["name"]].append(launch)
    metadata = json.loads((directory/"workload/profile_metadata.json").read_text())
    leading_name = max(by_name, key=lambda name: sum(event["end"]-event["start"] for event in by_name[name]))
    leading_id = next(identifier for identifier,name in strings.items() if name == leading_name)
    leading_geometry = [dict(grid=[gx,gy,gz],block=[bx,by,bz],dynamic_shared_bytes=shared,
                             count=count,mean_duration_ns=mean,sum_duration_ns=total)
                        for gx,gy,gz,bx,by,bz,shared,count,mean,total in db.execute(
                            "SELECT gridX,gridY,gridZ,blockX,blockY,blockZ,dynamicSharedMemory,"
                            "count(*),avg(end-start),sum(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL "
                            "WHERE demangledName=? GROUP BY gridX,gridY,gridZ,blockX,blockY,blockZ,"
                            "dynamicSharedMemory ORDER BY sum(end-start) DESC",(leading_id,))]
    leading_first = [dict(start_ns=start,duration_ns=elapsed,grid=[gx,gy,gz],block=[bx,by,bz],dynamic_shared_bytes=shared)
                     for start,elapsed,gx,gy,gz,bx,by,bz,shared in db.execute(
                         "SELECT start,end-start,gridX,gridY,gridZ,blockX,blockY,blockZ,dynamicSharedMemory "
                         "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName=? ORDER BY start LIMIT 20",(leading_id,))]
    result = {
        "source_sqlite": str(path), "source_sha256": hashlib.file_digest(path.open("rb"),"sha256").hexdigest(),
        "profiled": True, "headline_eligible": False,
        "measured_nvtx": {**bounds,"duration_ns": bounds["end"]-bounds["start"]},
        "metadata": metadata,
        "kernels": summary(kernels), "all_cuda_activity": summary(events),
        "families": {name:summary(group) for name,group in families.items()},
        "kernel_names": sorted(({"name":name,"family":family(name),**summary(group)}
                                 for name,group in by_name.items()),key=lambda row:row["sum_duration_ns"],reverse=True),
        "leading_kernel": {"name":leading_name,"geometry_groups":leading_geometry,"first_20_launches":leading_first},
        "transfers": {name:{**summary(group),"bytes":sum(event["bytes"] for event in group)}
                      for name,group in transfers.items()},
        "memset": {**summary([event for event in events if event["kind"] == "memset"]),
                   "bytes":sum(event.get("bytes",0) for event in events if event["kind"] == "memset")},
        "stages": {name:{"host_nvtx":next((stage for stage in stages if stage["name"] == name),None),
                         "all_cuda_activity":summary(group),
                         "kernels":summary([event for event in group if event["kind"] == "kernel"])}
                   for name,group in stage_events.items()},
        "api": {name:summary(group) for name,group in api_groups.items()},
        "gaps": {"internal_count":len(gaps), "internal_sum_ns":sum(end-start for start,end in gaps),
                 "initial_ns":initial_gap[1]-initial_gap[0],"trailing_ns":trailing_gap[1]-trailing_gap[0],
                 "max_internal_ns":max(end-start for start,end in gaps),
                 "internal_by_next_api_state_ns":dict(gap_classes),
                 "late_next_api_prefix_ns":late_launch_prefix,
                 "internal_over_1ms_count":sum(end-start>1_000_000 for start,end in gaps),
                 "internal_over_1ms_sum_ns":sum(end-start for start,end in gaps if end-start>1_000_000)},
        "method_notes": [
            "Family classification uses ordered name heuristics; it cannot establish compute or bandwidth saturation.",
            "Summed CUDA kernel duration is not wall time. Union durations account for overlap.",
            "Stage attribution uses correlated CUDA API entry within NVTX host ranges.",
            "CUDA API waits overlap GPU work; their summed duration is not additive overhead.",
            "Late API gap prefixes are dispatch-delay candidates, not proof of a CPU bottleneck; no CPU sampling was enabled.",
        ],
    }
    db.close()
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.directory)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"kernels":result["kernels"],
                      "all_cuda_activity":result["all_cuda_activity"],"gaps":result["gaps"]},indent=2))
