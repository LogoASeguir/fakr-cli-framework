from __future__ import annotations

from typing import Dict, Any, List


class MemoryMoment:
    """
    A single 'moment' in the runtime:

    - sensory: raw surface data (texts, signals)
    - internal: internal state (clocks, resonance, style, embryo weights, etc.)
    - context: design / task / environment metadata
    - perspective_depth: 0 = first-person, higher = observer layers
    """

    def __init__(
        self,
        sensory: Dict[str, Any],
        internal: Dict[str, Any],
        context: Dict[str, Any],
        perspective_depth: int = 0,
    ) -> None:
        self.sensory = sensory
        self.internal = internal
        self.context = context
        self.perspective_depth = perspective_depth

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sensory": self.sensory,
            "internal": self.internal,
            "context": self.context,
            "perspective_depth": self.perspective_depth,
        }


def observer_function(moment: MemoryMoment) -> MemoryMoment:
    """
    Creates a higher-perspective reconstruction of the moment.
    This is the core of MPM recursion.
    """
    # Copy internal state and add a simple 'observer_insight' flag.
    new_internal = dict(moment.internal)
    new_internal["observer_insight"] = True

    return MemoryMoment(
        sensory=moment.sensory,
        internal=new_internal,
        context=moment.context,
        perspective_depth=moment.perspective_depth + 1,
    )


def generate_perspective_stack(moment: MemoryMoment, max_depth: int = 3) -> List[MemoryMoment]:
    """
    Build a stack of increasingly higher-perspective versions of a moment.
    """
    stack: List[MemoryMoment] = [moment]
    for _ in range(max_depth):
        moment = observer_function(moment)
        stack.append(moment)
    return stack
