"""Reusable measurement and configuration-search interfaces."""

from gymnasium.envs.registration import register, registry

__version__ = "0.4.0"


if "DriveBench/OperatorTune-v0" not in registry:
    register(
        id="DriveBench/OperatorTune-v0",
        entry_point="drivebench_tuning.registration:OperatorTuneEnv",
        # Replay observations and rewards are seeded, but evidence keeps unique
        # event IDs across resets. Gymnasium's full-info determinism check would
        # otherwise require reusing those IDs.
        nondeterministic=True,
    )

if "DriveBench/KernelSchedule-v0" not in registry:
    register(
        id="DriveBench/KernelSchedule-v0",
        entry_point="drivebench_tuning.schedule_env:ScheduleTuningEnv",
        # Live CUDA timings and wall-clock budgets are intentionally stochastic.
        nondeterministic=True,
    )
