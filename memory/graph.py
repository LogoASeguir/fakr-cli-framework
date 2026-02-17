# memory/graph.py (or memory/skills.py)

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

from config import MEMORY_DIR
import json


@dataclass
class SkillCard:
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
    def __init__(self) -> None:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self.path = MEMORY_DIR / "skills.json"
        self.skills: Dict[str, SkillCard] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_text(encoding="utf-8")
        items = json.loads(raw)
        for obj in items:
            card = SkillCard.from_dict(obj)
            self.skills[card.id] = card

    def save(self) -> None:
        data = [card.to_dict() for card in self.skills.values()]
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def add_skill(self, card: SkillCard) -> None:
        self.skills[card.id] = card
        self.save()

    def find_by_keywords(self, text: str, top_k: int = 3) -> List[SkillCard]:
        """
        Super simple matching: count how many keywords appear in text.
        Later we can swap this for embeddings / vector search.
        """
        text_lower = text.lower()
        scored: List[tuple[int, SkillCard]] = []
        for card in self.skills.values():
            score = sum(1 for kw in card.context_keywords if kw.lower() in text_lower)
            if score > 0:
                scored.append((score, card))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [card for _, card in scored[:top_k]]
