# Research website

Static source for the public DriveBench research site. Serve this directory with a static HTTP server; no build, API keys, account or framework is required.

```sh
python -m http.server 8766 --directory website
```

The page states the active thesis and separates the legacy GPU evidence from the unrun modern-baseline, verifier, table, transfer and code-action phases. CPU examples live under software smoke tests, not empirical evidence.

The historical figures are unchanged numerical plots with downloadable data, captions, hashes and a reproducible generator in `figures/`. They are not modern-runtime measurements. The research dependency diagram is conceptual and explicitly labeled. No hosted evaluation service, completed lookup table, paper or leaderboard is implied.
