from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import json

from config import MEMORY_DIR


@dataclass
class ReasoningPattern:
    """
    Reusable 'way of thinking'.

    Example:
      id: "clarify_then_solve_v1"
      summary: "Ask 1–2 clarifying questions, then propose a concrete plan."
      template: "1) Ask: ... 2) Then do: ..."
      notes: "Where it's useful, caveats, etc."
    """
    id: str
    summary: str
    template: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "summary": self.summary,
            "template": self.template,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningPattern":
        return cls(
            id=data["id"],
            summary=data["summary"],
            template=data.get("template", ""),
            notes=data.get("notes", ""),
        )


class PatternStore:
    """
    Persistent library of ReasoningPattern objects, stored as patterns.json.
    """

    def __init__(self) -> None:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self.path: Path = MEMORY_DIR / "patterns.json"
        # Internal mapping: id -> ReasoningPattern
        self.patterns: Dict[str, ReasoningPattern] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_text(encoding="utf-8")
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return
        for obj in items:
            patt = ReasoningPattern.from_dict(obj)
            self.patterns[patt.id] = patt

    def save(self) -> None:
        data = [p.to_dict() for p in self.patterns.values()]
        self.path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def add_pattern(self, patt: ReasoningPattern) -> None:
        self.patterns[patt.id] = patt
        self.save()

    def get_pattern(self, pattern_id: str) -> ReasoningPattern | None:
        """
        Return a ReasoningPattern by id, or None if not found.
        """
        return self.patterns.get(pattern_id)

    def all_ids(self) -> list[str]:
        """
        Return a list of all stored pattern ids.
        """
        return list(self.patterns.keys())

    def find_relevant(self, text: str, limit: int = 3) -> list[ReasoningPattern]:
        """
        Very simple text match:
        - build a bag of words from pattern id + summary + template
        - count overlap with words in 'text'
        - return up to 'limit' patterns with highest overlap
        """
        import re

        tokens = {w.lower() for w in re.findall(r"\w+", text)}
        scored: list[tuple[int, ReasoningPattern]] = []

        for patt in self.patterns.values():
            haystack = f"{patt.id} {patt.summary} {patt.template}"
            words = {w.lower() for w in re.findall(r"\w+", haystack)}
            score = len(tokens & words)
            if score > 0:
                scored.append((score, patt))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [p for score, p in scored[:limit]]
