# DriveBench website

Static source for the public DriveBench benchmark and agent tuning environment. No build step, API keys or framework is required. From the repository root:

```sh
python -m http.server 8766 --bind 127.0.0.1 --directory website
```

Open <http://127.0.0.1:8766/>. The page includes measured operator results, policy comparisons, downloadable figures, and CPU replay and GPU collection examples. Dataset downloads and documentation link to their versioned repository paths.

The current operator figures in `figures/generalization/` are copied byte-for-byte from the published dataset. The page links their SVG and PDF versions for full-size viewing. Numerical data, measurement protocols and reproduction commands live with the dataset rather than in the website source. Historical figure files remain available in `figures/`.

`app.js` handles accessible example tabs and copy buttons. The site does not run GPU jobs or call model APIs; evaluation runs through the Python interfaces described in the documentation.
