# Registered-einops compiler discovery

On one development request, official einops operation registration, a native-attention Dynamo boundary and an explicit specialization cache limit of 512 produced 347 partial graphs and positive CUDA code-generation evidence. All four requests retained 347 graphs, with no recorded unimplemented-compiler fallback or OOM. Graph breaks remained; this was not full-graph compilation.

The cold request took 122.568308 s. Two warmups took 43.556760 and 43.588568 s; the one measured request took 43.607201 s. These synchronized preparation/generation times exclude file decode and output export. They do not establish a comparative speedup.

The unchanged zero-tolerance comparison **failed** on every execution: max absolute error 0.250361145 and RMSE 0.002511569 over 44,236,800 values. Perceptual quality remains unevaluated here; no tolerance was widened. The producing package was `2ead184559615d7c9c68620abca687431231ae9ea999a0189a39a0ab7a55206a`, with the helper and scoped runtime identities recorded separately.

[Numeric evidence and provenance](summary.json). The transfer receipt covers metadata and host logs; arrays and compiler binaries are excluded.
