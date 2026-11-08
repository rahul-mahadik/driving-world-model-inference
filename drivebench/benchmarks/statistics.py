import numpy as np


def bootstrap_quantile(values, quantile=.5, repeats=2000, seed=17, groups=None):
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return None
    rng = np.random.default_rng(seed)
    if groups is None:
        samples = rng.choice(x, size=(repeats, len(x)), replace=True)
        estimates = np.quantile(samples, quantile, axis=1)
    else:
        # Repeated requests from a scene are correlated; resample scenes, not rows.
        names = sorted(set(groups))
        if len(names) < 2:
            return None
        clusters = [x[np.asarray(groups) == name] for name in names]
        estimates = [np.quantile(np.concatenate([clusters[i] for i in rng.integers(0, len(names), len(names))]), quantile)
                     for _ in range(repeats)]
    return np.quantile(estimates, [.025, .975]).tolist()


def summarize(rows, total_wall_s, backend, headline=False, profiled=False):
    ok = [row for row in rows if row["status"] == "ok"]
    latencies = [row["latency_s"] for row in ok]
    scenes = [row["scene_id"] for row in ok]
    completed = sum(len(row.get("candidates", [])) for row in rows)
    oom_attempts = sum(len(row.get("cache_stats", {}).get("oom_attempts", [])) for row in rows)
    return {
        "backend": backend, "synthetic": backend == "synthetic", "profiled": profiled,
        "measured_requests": len(rows), "successful_requests": len(ok),
        "failures": len(rows) - len(ok), "completed_futures": completed,
        "oom_attempts": oom_attempts,
        "latency_p50_s": float(np.quantile(latencies, .5)) if latencies else None,
        "latency_p95_s": float(np.quantile(latencies, .95)) if len(latencies) >= 20 else None,
        "latency_p50_ci95_s": bootstrap_quantile(latencies, .5, groups=scenes),
        "latency_p95_ci95_s": bootstrap_quantile(latencies, .95, groups=scenes) if len(latencies) >= 20 else None,
        "uncertainty_method": "2000 scene-cluster bootstrap resamples; repeated scenes remain grouped",
        "throughput_futures_s": completed / total_wall_s if total_wall_s else None,
        "measurement_wall_s": total_wall_s,
        "peak_allocated_bytes": max((r.get("peak_allocated_bytes", 0) for r in rows), default=0),
        "peak_reserved_bytes": max((r.get("peak_reserved_bytes", 0) for r in rows), default=0),
        "headline_eligible": bool(headline and not profiled and backend != "synthetic" and len(ok) >= 200 and len(set(scenes)) >= 2 and len(ok) == len(rows) and not oom_attempts),
        "quality_preserving": False,
        "note": "Quality claims require separate evaluation against frozen development tolerances.",
    }
