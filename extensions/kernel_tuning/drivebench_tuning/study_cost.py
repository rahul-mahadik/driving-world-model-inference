"""Transparent request/storage arithmetic, not a spending admission or quotation."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def estimate(configurations, scenes, repeats, *, seconds_per_request=(45.,60.),
             compute_usd_per_hour=4.746, storage_usd_per_gb_month=.20,
             retention_days=30, raw_bytes_per_output=176947328, copies=2):
    for name,value in [('configurations',configurations),('scenes',scenes),('repeats',repeats),
                       ('raw_bytes_per_output',raw_bytes_per_output),('copies',copies)]:
        if type(value) is not int or value<1:raise ValueError(name+' must be a positive integer')
    rates=[*seconds_per_request,compute_usd_per_hour,storage_usd_per_gb_month,retention_days]
    if len(seconds_per_request)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in rates) or seconds_per_request[0]>seconds_per_request[1]:
        raise ValueError('finite nonnegative ordered timing and rate assumptions required')
    requests=configurations*scenes*repeats
    hours=[requests*value/3600 for value in seconds_per_request]
    raw=requests*raw_bytes_per_output
    return {'kind':'illustrative_capacity_and_cost_projection','resource_admitted':False,
            'configurations':configurations,'scenes':scenes,'repeats':repeats,
            'minimum_candidate_measurement_requests':requests,
            'seconds_per_request_assumption':list(seconds_per_request),
            'illustrative_gpu_hours':hours,
            'compute_usd_per_hour_assumption':compute_usd_per_hour,
            'illustrative_compute_usd':[h*compute_usd_per_hour for h in hours],
            'uncompressed_raw_bytes_per_output':raw_bytes_per_output,
            'raw_bytes_one_copy':raw,'raw_decimal_gb_one_copy':raw/1e9,
            'copies_for_capacity_planning':copies,'raw_bytes_all_copies':raw*copies,
            'storage_usd_per_decimal_gb_month_assumption':storage_usd_per_gb_month,
            'retention_days':retention_days,
            'illustrative_storage_usd_one_copy_30day_month':raw/1e9*storage_usd_per_gb_month*retention_days/30,
            'excluded':['reference generations','calibration and negative controls','first requests and warmups',
                        'compilation and setup','profiling','failed/partial attempts','quality metrics/human review',
                        'API policy cost','backups/transfer and verification time','cleanup and billing grace',
                        'artifact metadata and allocation capacity rounding'],
            'assumption_provenance':'Function defaults use historical H100 reference timing/rates; supplied overrides are caller assumptions, not fresh quotes or measurements.',
            'interpretation':'The request count is a lower bound for the specified Cartesian measurement grid. The default 45–60s time range is historical illustrative reference latency, NOT a lower bound or promise for lossy/modern configurations. Storage assumes full float32 arrays; compression savings are not assumed. Extra copies need separately priced locations.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configurations',type=int,default=1000);p.add_argument('--scenes',type=int,required=True)
    p.add_argument('--repeats',type=int,required=True);p.add_argument('--output',required=True)
    p.add_argument('--compute-usd-per-hour',type=float,default=4.746)
    p.add_argument('--seconds-per-request',type=float,nargs=2,default=[45.,60.],metavar=('LOW','HIGH'))
    p.add_argument('--storage-usd-per-gb-month',type=float,default=.20)
    p.add_argument('--retention-days',type=float,default=30)
    args=p.parse_args();result=estimate(args.configurations,args.scenes,args.repeats,
        seconds_per_request=tuple(args.seconds_per_request),compute_usd_per_hour=args.compute_usd_per_hour,
        storage_usd_per_gb_month=args.storage_usd_per_gb_month,retention_days=args.retention_days)
    with Path(args.output).open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')


if __name__=='__main__':main()
