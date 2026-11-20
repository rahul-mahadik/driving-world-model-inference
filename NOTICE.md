# Third-party notices and publication scope

First-party DriveBench code and documentation are licensed under the [MIT license](LICENSE), copyright 2026 Rahul Mahadik. This grant does not replace the licenses or terms of third-party dependencies, model weights or datasets.

Vista is developed by OpenDriveLab. The pinned source revision is `cc9821b4253ca7987c32757613d2fc2448fa9f5d`; its repository uses Apache License 2.0. An attribution copy is included at [third_party/Vista-LICENSE](third_party/Vista-LICENSE). The Vista source tree and checkpoint are obtained separately and are not redistributed here. The harness calls the upstream interfaces and pins provenance; Vista retains its own license.

nuScenes is provided by Motional. Obtain data and review its current [dataset terms](https://www.nuscenes.org/terms-of-use) and [download instructions](https://www.nuscenes.org/download) separately. This repository includes no dataset images, annotation tables, generated driving videos, or checkpoints. The metric-only chart is generated from measured numeric profiler summaries. No claim is made here about redistribution rights for omitted media.

PyTorch, xFormers, and other dependencies retain their respective licenses. Dependency definitions and versions are recorded. The sole bundled dependency is the unmodified pure-Python ANTLR4 runtime wheel described below; GPU libraries, model weights and datasets are obtained separately.

Portable publication files omit credentials, cloud project/instance IDs, SSH endpoints, machine-specific home paths, and account billing records. Generic provider API code and clearly marked test credentials remain as source code. This revision contains current study tooling. Initial evidence retains its original corrected-source identity and does not validate newer changes.

The small `third_party/wheels/antlr4_python3_runtime-4.9.3-py3-none-any.whl` preserves the exact dependency artifact used by the modern feasibility run. ANTLR4 is copyright The ANTLR Project and distributed under BSD-3-Clause; see [its complete upstream license](third_party/ANTLR4-LICENSE) and [source/build provenance](third_party/antlr4-wheel-provenance.json). The wheel contains no native libraries. Its license is separate from this repository's MIT license.
