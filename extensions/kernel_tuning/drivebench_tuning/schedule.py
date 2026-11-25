"""Immutable schedules and local edits for the experimental live CUDA track.

These are schedules for fixed operator code, not arbitrary source submissions.
Inactive fields have canonical values so they cannot create duplicate actions.
The earlier finite configuration table and its action numbering are unchanged.
"""
from dataclasses import asdict, dataclass, replace
from numbers import Integral

from .contract import Workload

BLOCKS = (64, 128, 256, 512, 1024, 2048, 4096)
WARPS = (1, 2, 4, 8)


@dataclass(frozen=True)
class KernelSchedule:
    fused: bool = True
    block_size: int = 256
    num_warps: int = 4
    activation_block_size: int = 256
    activation_num_warps: int = 4
    reduction: str = "chunked"
    reduction_block_size: int = 256
    reduction_num_warps: int = 4

    def __post_init__(self):
        if type(self.fused) is not bool:
            raise ValueError("fused must be a boolean")
        for field in ("block_size", "activation_block_size", "reduction_block_size"):
            if type(getattr(self, field)) is not int or getattr(self, field) not in BLOCKS:
                raise ValueError(f"{field} must be one of {BLOCKS}")
        for field in ("num_warps", "activation_num_warps", "reduction_num_warps"):
            if type(getattr(self, field)) is not int or getattr(self, field) not in WARPS:
                raise ValueError(f"{field} must be one of {WARPS}")
        if self.reduction not in ("chunked", "single"):
            raise ValueError("reduction must be chunked or single")

    def to_dict(self):
        return asdict(self)


# Action indices are part of this track's versioned contract.
ACTIONS = (
    "block_down", "block_up", "warps_down", "warps_up",
    "unfuse", "fuse",
    "activation_block_down", "activation_block_up",
    "activation_warps_down", "activation_warps_up",
    "reduction_single", "reduction_chunked",
    "reduction_block_down", "reduction_block_up",
    "reduction_warps_down", "reduction_warps_up",
    "measure", "submit",
)


def check_schedule(workload: Workload, schedule: KernelSchedule):
    if not isinstance(workload, Workload) or not isinstance(schedule, KernelSchedule):
        raise TypeError("a Workload and KernelSchedule are required")
    if schedule.fused and (schedule.activation_block_size, schedule.activation_num_warps) != (256, 4):
        raise ValueError("fused schedules require canonical inactive activation fields")
    if workload.operator == "geglu":
        if (schedule.reduction, schedule.reduction_block_size, schedule.reduction_num_warps) != ("chunked", 256, 4):
            raise ValueError("GEGLU has no reduction schedule")
    else:
        from math import prod
        size = prod(workload.shape[1:]) // workload.num_groups
        if schedule.reduction == "single":
            if schedule.reduction_block_size != 256:
                raise ValueError("single reduction requires canonical inactive block size")
            if size > 65536:
                raise ValueError("single reduction supports at most 65536 elements per group")
        elif (size + schedule.reduction_block_size - 1) // schedule.reduction_block_size > 4096:
            raise ValueError("chunked reduction supports at most 4096 chunks per group")


def initial_schedule(workload: Workload) -> KernelSchedule:
    """Choose a fixed legal start without querying any performance outcome."""
    for block in BLOCKS[BLOCKS.index(256):]:
        schedule = KernelSchedule(reduction_block_size=block) if workload.operator == "groupnorm_silu" else KernelSchedule()
        try:
            check_schedule(workload, schedule)
            return schedule
        except ValueError:
            continue
    raise ValueError("workload exceeds the supported reduction schedule range")


def transform(workload: Workload, schedule: KernelSchedule, action: int | str) -> KernelSchedule:
    """Apply one local edit; illegal edits raise without changing the input.

    Unfusing copies the current main launch into a new activation stage. Fusing
    removes that stage and its launch choices. Editing then unfusing therefore
    differs from unfusing then editing; actions operate on the current program.
    """
    check_schedule(workload, schedule)
    if isinstance(action, Integral) and not isinstance(action, bool):
        if not 0 <= action < len(ACTIONS):
            raise ValueError("action index out of range")
        action = ACTIONS[int(action)]
    if not isinstance(action, str) or action not in ACTIONS:
        raise ValueError("unknown schedule action")
    if action in ("measure", "submit"):
        return schedule
    if action == "unfuse":
        if not schedule.fused:
            raise ValueError("schedule is already unfused")
        result = replace(schedule, fused=False, activation_block_size=schedule.block_size,
                         activation_num_warps=schedule.num_warps)
    elif action == "fuse":
        if schedule.fused:
            raise ValueError("schedule is already fused")
        result = replace(schedule, fused=True, activation_block_size=256, activation_num_warps=4)
    elif action in ("reduction_single", "reduction_chunked"):
        if workload.operator != "groupnorm_silu":
            raise ValueError("operator has no reduction")
        reduction = action.removeprefix("reduction_")
        if schedule.reduction == reduction:
            raise ValueError("reduction strategy is already selected")
        result = replace(schedule, reduction=reduction, reduction_block_size=256)
    else:
        key, direction = action.rsplit("_", 1)
        field = {"block": "block_size", "warps": "num_warps",
                 "activation_block": "activation_block_size", "activation_warps": "activation_num_warps",
                 "reduction_block": "reduction_block_size", "reduction_warps": "reduction_num_warps"}[key]
        if key.startswith("activation") and schedule.fused:
            raise ValueError("unfuse before editing the activation stage")
        if key.startswith("reduction") and workload.operator != "groupnorm_silu":
            raise ValueError("operator has no reduction")
        if key == "reduction_block" and schedule.reduction == "single":
            raise ValueError("single reduction has no chunk size")
        choices = BLOCKS if "block" in key else WARPS
        index = choices.index(getattr(schedule, field)) + (1 if direction == "up" else -1)
        if not 0 <= index < len(choices):
            raise ValueError("edit exceeds the supported schedule range")
        result = replace(schedule, **{field: choices[index]})
    check_schedule(workload, result)
    return result


def action_mask(workload: Workload, schedule: KernelSchedule) -> list[int]:
    mask = []
    for index in range(len(ACTIONS)):
        try:
            transform(workload, schedule, index)
            mask.append(1)
        except ValueError:
            mask.append(0)
    return mask
