# Canceled prospective studies

On November 14, 2026, the project direction changed to **verifiable rewards for lossy inference optimization of video world models**.

The planned one-episode, eight-configuration LLM/random/grid Vista study was canceled **before execution**. It cannot support the repeated optimizer or transfer questions now prioritized. Do not launch `configs/research/vista-three-policy-v1.json` or treat the old feasibility recipe as the active study.

The original plan files, [former research protocol](archive/research-protocol-v1.md), private preparations, failed attempts and historical measurements remain preserved. Their presence does not authorize execution and their original `execution_performed: false` state is unchanged. The old fixed MAE/temporal/LPIPS limits are historical engineering proxies; they are not empirically calibrated gates for the new program.

The [current protocol](research-protocol.md) and [machine-readable registry](../configs/research/active-roadmap-v2.json) supersede those plans. New execution needs a separate prospectively frozen configuration, source/runtime/data bindings, cost/storage plan and immutable resource deadline. Existing bounded measurement and preservation primitives may be reused; historical bindings and deadlines may not.

More legacy eager-versus-compiled repeats and CPU synthetic demos as empirical evidence are excluded from the new program. Existing synthetic tests remain software checks, not research results.
