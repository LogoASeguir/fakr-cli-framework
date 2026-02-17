from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any

from config import DESIGNS_DIR


@dataclass
class Turn:
    """
    One step in the design conversation.

    role: "user", "assistant", or later maybe "system"/"critic"
    text: raw text of the message
    """
    role: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "text": self.text}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        return cls(role=data.get("role", "user"), text=data.get("text", ""))


@dataclass
class DesignState:
    """
    Persistent record of a single design "thread".

    - id: stable identifier (UUID string)
    - goal: short description of what this design is trying to achieve
    - turns: full path of the conversation so far
    """
    id: str
    goal: str
    turns: List[Turn] = field(default_factory=list)

    @property
    def path(self) -> Path:
        """
        Where this design is stored on disk.
        """
        DESIGNS_DIR.mkdir(parents=True, exist_ok=True)
        return DESIGNS_DIR / f"{self.id}.json"

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        """
        Persist this design to disk as JSON.
        """
        data = {
            "id": self.id,
            "goal": self.goal,
            "turns": [t.to_dict() for t in self.turns],
        }
        self.path.write_text(
            __import__("json").dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, design_id: str) -> "DesignState":
        """
        Load a design by id from DESIGNS_DIR.
        """
        path = DESIGNS_DIR / f"{design_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No design file found for id={design_id}")

        raw = path.read_text(encoding="utf-8")
        data = __import__("json").loads(raw)

        turns = [Turn.from_dict(t) for t in data.get("turns", [])]
        return cls(
            id=data["id"],
            goal=data["goal"],
            turns=turns,
        )

    # ------------------------------------------------------------------ #
    # Conversation / path helpers
    # ------------------------------------------------------------------ #

    def record_turn(self, role: str, text: str) -> None:
        """
        Append a new turn to the design, but do NOT auto-save.
        (Caller decides when to save.)
        """
        self.turns.append(Turn(role=role, text=text))

    def summary(self) -> str:
        """
        Quick human-readable summary. You can use this later as context
        or for offline consolidation.
        """
        return f"Design {self.id} – goal: {self.goal}, turns={len(self.turns)}"
