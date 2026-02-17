from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class ClockTier(str, Enum):
    """
    Coarse time scales inside the runtime.

    - FAST   – every user interaction
    - MEDIUM – every N interactions
    - SLOW   – every M interactions
    """
    FAST = "FAST"
    MEDIUM = "MEDIUM"
    SLOW = "SLOW"


@dataclass
class ClockConfig:
    name: ClockTier
    interval: int            # how many steps between ticks
    last_tick_step: int = 0  # last global step when this clock fired


class ClockState:
    """
    Tracks an internal 'step' counter and decides which clocks fire
    on each user interaction.
    """

    def __init__(self) -> None:
        self.step: int = 0
        self.clocks: Dict[ClockTier, ClockConfig] = {}

    def register_clock(self, tier: ClockTier, interval: int) -> None:
        """
        Register or overwrite a clock tier with a given interval.

        interval = 1  -> fires every step
        interval = 5  -> fires every 5th step
        """
        self.clocks[tier] = ClockConfig(name=tier, interval=interval)

    def advance(self) -> List[ClockTier]:
        """
        Advance the global step by 1 and return the list of clock tiers
        that should fire on this step.
        """
        self.step += 1
        fired: List[ClockTier] = []

        for cfg in self.clocks.values():
            if cfg.interval <= 0:
                # Disabled / invalid clock
                continue

            if (self.step - cfg.last_tick_step) >= cfg.interval:
                cfg.last_tick_step = self.step
                fired.append(cfg.name)

        return fired


# ---------------------------------------------------------------------- #
# Multi-perspective time dilation: mapping depth → time-scale factor
# ---------------------------------------------------------------------- #

PERSPECTIVE_CLOCKS: Dict[int, float] = {
    # depth 0 → present
    0: 0.0,
    # depth 1 → short-term reconstruction
    1: 0.25,
    # depth 2 → reflective meta-awareness
    2: 1.0,
    # depth 3 → long-term narrative compression
    3: 4.0,
}
