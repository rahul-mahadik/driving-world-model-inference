# Sequential schedule comparison

A small exploratory comparison of a fixed reference, random schedule edits, and greedy local search on six task families, with three seeds per family.

| Policy | Primary score (95% interval) | Confirmed | Actual episode time | Query-cap hits |
|---|---:|---:|---:|---:|
| compiled_eager_no_edit | -0.0306 [-0.1453, 0.0629] | 18/18 | 135.7 s | 0 |
| random_walk | -0.4346 [-0.7758, -0.1075] | 12/18 | 823.3 s | 3 |
| greedy_local | -0.1640 [-0.3700, 0.0055] | 18/18 | 827.2 s | 6 |

Confirmed episodes: clip(1 - median(selected confirmation samples) / median(paired final reference samples), -1, 1). Failed or unconfirmed episodes: -1. The reference role is selected during reset and fixed before final measurement. This score differs from the environment return, which uses its initial reference.

Intervals resample six task families, then the three seeds within each family, using identical draws for all policies. This is a small exploratory study. It does not evaluate a trained RL policy, measure an oracle, or establish transfer to unseen operators.

Actual elapsed time per episode includes reference setup, compilation, policy work, failures, confirmation and teardown; sums exclude gaps between episodes and outer host setup/validation.

All 54 attempts, including failed and unconfirmed episodes, appear in the JSON and CSV exports. Successful-only latency ratios are secondary; the primary score includes failures.
