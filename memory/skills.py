from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import json

from config import MEMORY_DIR


@dataclass
class SkillCard:
    """
    Frozen solution / design snippet.

    Example:
      id: "python_cli_calculator_v1"
      summary: "Minimal Python CLI calculator with + - * /."
      context_keywords: ["python", "cli", "calculator"]
      core_code: "...code snippet..."
      notes: "Anything extra."
    """
    id: str
    summary: str
    context_keywords: List[str]
    core_code: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "summary": self.summary,
            "context_keywords": self.context_keywords,
            "core_code": self.core_code,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillCard":
        return cls(
            id=data["id"],
            summary=data["summary"],
            context_keywords=data.get("context_keywords", []),
            core_code=data.get("core_code", ""),
            notes=data.get("notes", ""),
        )


class SkillStore:
    """
    Persistent library of SkillCard objects, stored as skills.json in MEMORY_DIR.
    """

    def __init__(self) -> None:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self.path: Path = MEMORY_DIR / "skills.json"
        # Internal mapping: id -> SkillCard
        self.skills: Dict[str, SkillCard] = {}
        self._load()

    # ---------------- internal persistence ---------------- #

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_text(encoding="utf-8")
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            # If file is corrupt, ignore for now
            return
        for obj in items:
            card = SkillCard.from_dict(obj)
            self.skills[card.id] = card

    def save(self) -> None:
        data = [card.to_dict() for card in self.skills.values()]
        self.path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ---------------- public API ---------------- #

    def add_skill(self, card: SkillCard) -> None:
        """
        Add or overwrite a skill and persist.
        """
        self.skills[card.id] = card
        self.save()

    def get_skill(self, skill_id: str) -> SkillCard | None:
        """
        Return a SkillCard by id, or None if not found.
        """
        return self.skills.get(skill_id)

    def all_ids(self) -> list[str]:
        """
        Return a list of all stored skill ids.
        """
        return list(self.skills.keys())

    def find_relevant(self, text: str, limit: int = 3) -> list[SkillCard]:
        """
        Very simple keyword match:
        - look at context_keywords for each skill
        - count overlap with words in 'text'
        - return up to 'limit' skills with highest overlap
        """
        import re

        tokens = {w.lower() for w in re.findall(r"\w+", text)}
        scored: list[tuple[int, SkillCard]] = []

        for card in self.skills.values():
            kws = {k.lower() for k in getattr(card, "context_keywords", [])}
            score = len(tokens & kws)
            if score > 0:
                scored.append((score, card))

        # Sort by score descending
        scored.sort(key=lambda t: t[0], reverse=True)
        return [card for score, card in scored[:limit]]
