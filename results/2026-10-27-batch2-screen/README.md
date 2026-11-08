# Two-candidate joint-denoising screen

The real GPU execution used batch size `[2]`, with 50 denoiser calls and no OOM or fallback. The upstream reference used serial `[1,1]` execution and 100 calls. The unchanged zero-tolerance agreement check **failed**: candidate RMSE values were 0.003043780 and 0.003841399, with maximum absolute errors 0.202441037 and 0.466292858.

This establishes that joint execution was exercised and differs numerically. It establishes neither perceptual acceptability nor a performance gain; no tolerance was widened. The producing package was `3fd88646ba207e5e56c67317bd6b51c7f58f7da3c38ac3d9b739f2e90dee597b`, while the current published package is `d8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb`.

[Numeric comparison and provenance](summary.json). Metadata transfer hashes are verified; raw arrays are excluded.
