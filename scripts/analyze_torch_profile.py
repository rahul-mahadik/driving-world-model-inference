#!/usr/bin/env python3
"""Stream a PyTorch Chrome trace for CPU operator shapes and allocation events.

This does not load the whole JSON or infer memory bandwidth from allocation
bytes. Live allocated/reserved values come from CUDA allocator event fields;
positive allocation totals are cumulative churn, not live memory or DRAM IO.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path

import ijson


def analyze(path):
    counts = Counter()
    stages = []
    shape_ops = {"aten::copy_", "aten::contiguous", "aten::_to_copy", "aten::clone", "aten::addmm",
                 "aten::native_group_norm", "aten::native_layer_norm"}
    shapes = {name: Counter() for name in shape_ops}
    memory = defaultdict(lambda: {"count":0,"allocations":0,"frees":0,"allocated_bytes_sum":0,
                                  "freed_bytes_sum":0,"max_live_allocated":0,"max_reserved":0})
    sizes = Counter()
    top_allocations = []
    peak = None
    peak_by_stage = {}
    first = last = None
    nonmonotonic = 0
    cuda_events = 0
    categories = Counter()
    autocast_cache_clear_ranges = []
    outside_stage_frees = []
    def stage_for(ts):
        owners = [stage for stage in stages if stage["start_us"] <= ts < stage["end_us"]]
        return min(owners,key=lambda stage:stage["end_us"]-stage["start_us"])["name"] if owners else "outside_stage"

    with path.open("rb") as source:
        for event in ijson.items(source,"traceEvents.item",use_float=True):
            name = event.get("name","")
            category = event.get("cat","")
            categories[category] += 1
            counts[name] += 1
            args = event.get("args",{})
            if "clear_autocast_cache" in name and "dur" in event:
                autocast_cache_clear_ranges.append({"name":name,"category":category,
                                                    "start_us":event["ts"],"end_us":event["ts"]+event["dur"]})
            if category == "user_annotation" and name.startswith("drivebench::") and name != "drivebench::gpu_inference":
                stages.append({"name":name,"start_us":event["ts"],"end_us":event["ts"]+event["dur"]})
            if category == "cpu_op" and name in shapes:
                key = json.dumps({"input_dims":args.get("Input Dims"),"input_types":args.get("Input type")},sort_keys=True)
                shapes[name][key] += 1
            if name != "[memory]" or args.get("Device Type") != 1:
                continue
            cuda_events += 1
            ts = event["ts"]
            size = args.get("Bytes",0)
            allocated = args.get("Total Allocated",0)
            reserved = args.get("Total Reserved",0)
            stage = stage_for(ts)
            row = {"ts_us":ts,"bytes":size,"total_allocated":allocated,"total_reserved":reserved,"stage":stage}
            if first is None:
                first = row
            if last is not None and ts < last["ts_us"]:
                nonmonotonic += 1
            last = row
            if peak is None or allocated > peak["total_allocated"]:
                peak = row
            if stage not in peak_by_stage or allocated > peak_by_stage[stage]["total_allocated"]:
                peak_by_stage[stage] = row
            group = memory[stage]
            group["count"] += 1
            group["allocations"] += int(size > 0)
            group["frees"] += int(size < 0)
            group["allocated_bytes_sum"] += max(size,0)
            group["freed_bytes_sum"] += max(-size,0)
            group["max_live_allocated"] = max(group["max_live_allocated"],allocated)
            group["max_reserved"] = max(group["max_reserved"],reserved)
            if stage == "outside_stage" and size < 0:
                outside_stage_frees.append((ts,-size))
            if size > 0:
                sizes[size] += 1
                item = (size,ts,cuda_events,row)
                if len(top_allocations) < 20:
                    heapq.heappush(top_allocations,item)
                elif item[:3] > top_allocations[0][:3]:
                    heapq.heapreplace(top_allocations,item)
    if nonmonotonic:
        raise ValueError("Memory events are not timestamp sorted; first/last and live baseline require a sorted pass")
    for interval in autocast_cache_clear_ranges:
        matching=[size for ts,size in outside_stage_frees if interval["start_us"] <= ts <= interval["end_us"]]
        interval["outside_stage_cuda_frees"] = len(matching)
        interval["outside_stage_cuda_freed_bytes"] = sum(matching)
    # Find the CPU nesting at the peak allocation, without retaining millions of
    # operator or Python stack records in memory.
    enclosing_by_stage = {stage: [] for stage in peak_by_stage}
    peak_timestamps = sorted((row["ts_us"],stage) for stage,row in peak_by_stage.items())
    timestamp_values = [item[0] for item in peak_timestamps]
    with path.open("rb") as source:
        for event in ijson.items(source,"traceEvents.item",use_float=True):
            if event.get("cat") != "cpu_op":
                if event.get("name") == "[memory]":
                    break
                continue
            start=event.get("ts",0); end=start+event.get("dur",0)
            for _,stage in peak_timestamps[bisect_left(timestamp_values,start):bisect_right(timestamp_values,end)]:
                enclosing_by_stage[stage].append({"name":event["name"],"start_us":start,"duration_us":event.get("dur"),
                                                  "args":event.get("args",{})})
    return {
        "source":str(path),"source_sha256":hashlib.file_digest(path.open("rb"),"sha256").hexdigest(),
        "profiled":True,"headline_eligible":False,"event_category_counts":dict(categories),
        "cuda_memory_event_count":cuda_events,"cuda_memory_by_host_stage":dict(memory),
        "first_cuda_memory_event":first,"last_cuda_memory_event":last,
        "baseline_live_before_first_event":first["total_allocated"]-first["bytes"],
        "peak_cuda_memory_event":peak,"cpu_ops_enclosing_peak_allocation":enclosing_by_stage[peak["stage"]],
        "peak_cuda_memory_event_by_stage":peak_by_stage,
        "cpu_ops_enclosing_peak_allocation_by_stage":enclosing_by_stage,
        "autocast_cache_clear_ranges":autocast_cache_clear_ranges,
        "largest_allocation_events": [item[3] for item in sorted(top_allocations,reverse=True)],
        "allocation_size_groups_by_cumulative_bytes": [dict(bytes=size,count=count,cumulative_bytes=size*count)
            for size,count in sorted(sizes.items(),key=lambda item:item[0]*item[1],reverse=True)[:30]],
        "operator_shape_counts":{name:[dict(json.loads(shape),count=count) for shape,count in counter.most_common(15)]
                                 for name,counter in shapes.items()},
        "operator_counts":{name:counts[name] for name in sorted(shape_ops)},
        "method_notes":["Memory classification is by host allocation timestamp within stage NVTX ranges.",
                        "Allocation bytes are cumulative allocator bookkeeping, not live peak or DRAM traffic.",
                        "Shape counts have no per-shape CUDA duration and cannot rank per-shape GPU cost.",
                        "Operator table rows include overlapping CPU/operator/kernel/runtime levels; do not sum them."]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.trace)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:result[key] for key in ["event_category_counts","cuda_memory_event_count",
                      "baseline_live_before_first_event","peak_cuda_memory_event","cuda_memory_by_host_stage"]},indent=2))
