"""Non-overlapping timeline views from a separate diagnostic Chrome trace.

Each lane is a partition of one trace window. Different lanes are different
views of the SAME time, and are never stacked or added to request latency.
CPU operator nesting is resolved to the deepest active interval per thread;
concurrent work on multiple threads or devices gets an explicit overlap bin.
GPU kernels are shown by recorded kernel name, not guessed high-level operators.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

from drivebench.config import file_sha256

STAGES = {'conditioning_gpu', 'encode_gpu', 'denoising_gpu', 'decode_gpu',
          'upstream_sample_gpu', 'device_to_host_gpu'}


def partition(events, start_us, end_us, *, nested=False):
    """Integrate disjoint elementary intervals; never sum inclusive durations."""
    if not all(type(v) in (int,float) and math.isfinite(v) for v in (start_us,end_us)) or end_us <= start_us:
        raise ValueError('a finite positive trace window is required')
    clipped = []
    for event in events:
        begin, end = event['start_us'], event['end_us']
        if not all(type(v) in (int,float) and math.isfinite(v) for v in (begin,end)) or end < begin:
            raise ValueError('invalid event interval')
        if not isinstance(event.get('name'),str) or not event['name']:
            raise ValueError('event name required')
        begin, end = max(start_us,begin), min(end_us,end)
        if end > begin: clipped.append({**event,'start_us':begin,'end_us':end})
    boundaries = defaultdict(list)
    for index,event in enumerate(clipped):
        boundaries[event['start_us']].append((1,index))
        boundaries[event['end_us']].append((-1,index))
    active = defaultdict(dict); labels = {}; totals = defaultdict(float)
    previous = start_us
    for point in sorted({start_us,end_us,*boundaries}):
        present = [value for group in labels.values() for value in group]
        if not present: label = 'unassigned_or_idle'
        elif len(set(present)) == 1: label = present[0][1]
        else: label = 'overlapping_or_ambiguous'
        totals[label] += point-previous
        affected = set()
        for action,index in boundaries[point]:
            event=clipped[index];resource=event.get('resource');affected.add(resource)
            if action==1: active[resource][index]=event
            else: active[resource].pop(index)
        for resource in affected:
            group=list(active[resource].values())
            if nested:
                group=[event for event in group if not any(
                    event['start_us']<=other['start_us'] and event['end_us']>=other['end_us'] and
                    (event['start_us'],event['end_us'])!=(other['start_us'],other['end_us'])
                    for other in group)]
            labels[resource]=[(resource,event['name']) for event in group]
        previous=point
    totals={name:value for name,value in totals.items() if value}
    return {'window_us':end_us-start_us,'partition_us':dict(sorted(totals.items())),
            'partition_sum_us':sum(totals.values()),'exclusive_partition':True}


def analyze_trace(path, *, start_us, end_us):
    """Window must be the explicit profiled request range, not the whole process.

    Stream input to avoid holding an entire multi-million-event trace. No GPU
    time is attributed through CPU timestamps or an unverified correlation ID.
    """
    import ijson
    stages, cpu, device = [], [], []
    with Path(path).open('rb') as stream:
        for event in ijson.items(stream, 'traceEvents.item', use_float=True):
            if event.get('ph') != 'X' or 'ts' not in event or 'dur' not in event: continue
            start, duration = event['ts'], event['dur']
            if type(start) not in (int,float) or type(duration) not in (int,float) or not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
                raise ValueError('invalid complete event timestamp')
            end = start+duration
            if end <= start_us or start >= end_us: continue
            name, category = event.get('name',''), event.get('cat','')
            row = {'name':name,'start_us':start,'end_us':end,
                   'resource':str(event.get('pid'))+':'+str(event.get('tid'))}
            if category == 'user_annotation' and name.startswith('drivebench::') and name.split('::',1)[1] in STAGES:
                stages.append(row)
            elif category == 'cpu_op': cpu.append(row)
            elif category in ('kernel','gpu_memcpy','gpu_memset'): device.append(row)
    return {'schema_version':1,'trace_sha256':file_sha256(path),'profiled':True,'headline_eligible':False,
            'window':{'start_us':start_us,'end_us':end_us},
            'lanes':{'host_stage_ranges':partition(stages,start_us,end_us,nested=True),
                     'host_operator_exclusive_ranges':partition(cpu,start_us,end_us,nested=True),
                     'device_kernel_and_transfer_activity':partition(device,start_us,end_us)},
            'notes':['Three separate views, never add lanes or stack them into one latency total.',
                     'Stage ranges measure host submission intervals; they are not GPU stage latency.',
                     'Device names are recorded kernels/transfers; high-level operator attribution is not inferred.',
                     'Concurrent or ambiguous intervals occupy one explicit overlap bin, not each contributing operation.',
                     'Unassigned/idle means this trace view lacks a named event; it does not prove device idleness.',
                     'A separate profiled request is diagnostic and cannot replace unprofiled timing.']}


def render(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,(name,lane) in zip(axes,report['lanes'].items()):
        rows = sorted(lane['partition_us'].items(),key=lambda item:item[1],reverse=True)
        keep = rows[:9]; rest = sum(value for _,value in rows[9:])
        if rest: keep.append(('other_disjoint_intervals',rest))
        ax.barh([key for key,_ in reversed(keep)],[value/1000 for _,value in reversed(keep)])
        ax.set_xlabel('Exclusive elapsed time (ms)')
        ax.set_title(name.replace('_',' '),fontsize=10)
    fig.suptitle('Diagnostic timeline views — do not add panels')
    fig.savefig(output,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',required=True)
    p.add_argument('--start-us',required=True,type=float);p.add_argument('--end-us',required=True,type=float)
    p.add_argument('--output',required=True);p.add_argument('--figure')
    args=p.parse_args();result=analyze_trace(args.trace,start_us=args.start_us,end_us=args.end_us)
    with Path(args.output).open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    if args.figure:
        if Path(args.figure).exists():raise FileExistsError('refuse to overwrite attribution figure')
        render(result,args.figure)


if __name__ == '__main__':main()
