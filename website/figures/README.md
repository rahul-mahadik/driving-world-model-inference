# Historical H100 figures

These three descriptive figures are generated from the 400 actual public
request-level timing records in the V22 release. They introduce no new GPU
measurements, quality results, hardware cohorts or uncertainty estimates.

| Figure stem | Content |
| --- | --- |
| `h100-latency-ecdf` | Role-wise empirical latency distributions: a shared axis and explicitly labeled per-role zooms. |
| `h100-latency-memory` | Zero-based comparisons of median latency and peak CUDA allocated memory. |
| `h100-block-chronology` | The eight chronological block medians and descriptive deviations from each role's overall median. |

Each stem is available as SVG, PNG and PDF. `captions.json` supplies the full
scientific captions and alternative text. Use the captions with the figures:
the figures are historical performance evidence, and output quality remains
unvalidated. `plot-data.json` supplies every ECDF point and aggregate/block
coordinate. `plot-data.csv` contains the two role aggregates and eight block
summaries. Individual ECDF points retain public scene, block and request
indices, so they can be traced to the public source rows.

## Reproduce

Use the public repository at commit
`da39fd8e40a0d337e392fff1006920c7f6608676`, or a later checkout that preserves
the six source files pinned by `generate_figures.py`. The generator refuses
different input bytes, reconstructs all 48 recorded historical summary fields
using the pinned public analyzer, and only then plots the actual observations.
It also verifies that the pinned inputs remain unchanged after generation.

The verified plotting environment is Python 3.13.1, Matplotlib 3.10.0,
NumPy 2.2.0 and Pillow 11.0.0, using Matplotlib's bundled DejaVu Sans font.
An isolated Python environment with these versions can run:

```sh
python -B generate_figures.py --repository /path/to/public-checkout --output figures-reproduced
```

The output directory must be new. The generator never overwrites an earlier
artifact directory. No GPU, model files, credentials, downloads or network
access are needed. `figure-manifest.json` records input and output SHA-256
hashes, the generator hash and the plotting environment. Two clean runs in
the verified environment produced identical bytes for all 13 output files,
including the manifest. A different plotting library or font version may
change image bytes even when numerical coordinates remain identical.

## Interpretation

There are 200 requests per role: the same 50 scene groups repeated four
times, on one physical H100 and runtime. The 400 requests are not 400
independent scenes. There are no confidence bands or error bars in these
figures. The historical confidence intervals can instead be reconstructed
with the public analyzer's recorded scene-cluster bootstrap.

The role order is U,C,C,U,C,U,U,C, with 50 requests per block. The block axis
is chronological process launch order, not elapsed time; the figure does
not establish stationarity. Request latency includes input loading and
preprocessing through host-ready frames, excluding initialization,
compilation, warmups and export. Memory is the maximum recorded CUDA
allocated-memory peak, divided by 2^30 for GiB, rather than reserved memory
or total device memory.

This is a latency–memory tradeoff: median latency is 23.85% lower and peak
allocated memory is 7.68% higher for the compiled denoiser. The separately
reported sequential-throughput increase is 31.34%, computed from total
measured request time rather than the reciprocal of median latency. None
of these observations establishes quality preservation, a jointly validated
quality/performance operating point, concurrent-serving capacity, or
performance on another GPU.
