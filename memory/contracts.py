from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import uuid

from config import MEMORY_DIR

CONTRACTS_PATH = MEMORY_DIR / "contracts.json"


@dataclass
class Contract:
    id: str
    title: str
    description: str
    enabled: bool = True
    tags: List[str] | None = None


class ContractStore:
    """
    Soft workflow rules (not hard enforcement). These are appended to model context
    to reduce drift (e.g. 'don't output code when user asked to rephrase').
    """

    def __init__(self, path: Path = CONTRACTS_PATH) -> None:
        self.path = path
        self._contracts: Dict[str, Contract] = {}
        self._load()

        # If empty, seed defaults (nice for first run)
        if not self._contracts:
            self.reset_defaults()

    # ---------------------------- I/O ----------------------------

    def _load(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            return

        try:
            raw = self.path.read_text(encoding="utf-8").strip()
            data = json.loads(raw) if raw else {}
        except Exception:
            data = {}

        items = data.get("contracts", []) if isinstance(data, dict) else []
        for c in items:
            try:
                # Back-compat: older versions used "body"
                desc = c.get("description", None)
                if desc is None:
                    desc = c.get("body", "")

                self._contracts[c["id"]] = Contract(
                    id=str(c["id"]),
                    title=str(c.get("title", "")).strip(),
                    description=str(desc or "").strip(),
                    enabled=bool(c.get("enabled", True)),
                    tags=list(c.get("tags", []) or []),
                )
            except Exception:
                continue

    def _save(self) -> None:
        payload: Dict[str, Any] = {
            "contracts": [
                {
                    "id": c.id,
                    "title": c.title,
                    "description": c.description,
                    "enabled": c.enabled,
                    "tags": c.tags or [],
                }
                for c in self._contracts.values()
            ]
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # -------------------------- API -----------------------------

    def all_ids(self) -> List[str]:
        return sorted(self._contracts.keys())

    def get(self, cid: str) -> Optional[Contract]:
        return self._contracts.get(cid)

    def active(self) -> List[Contract]:
        # stable order: enabled first, then title
        return sorted([c for c in self._contracts.values() if c.enabled], key=lambda x: x.title.lower())

    def add(
        self,
        title: str,
        description: str,
        tags: Optional[List[str]] = None,
        enabled: bool = True,
    ) -> Contract:
        cid = str(uuid.uuid4())[:8]
        c = Contract(
            id=cid,
            title=title.strip(),
            description=description.strip(),
            enabled=enabled,
            tags=tags or [],
        )
        self._contracts[c.id] = c
        self._save()
        return c

    def set_enabled(self, cid: str, enabled: bool) -> bool:
        c = self._contracts.get(cid)
        if not c:
            return False
        c.enabled = bool(enabled)
        self._save()
        return True

    def reset_defaults(self) -> None:
        self._contracts = {}
        defaults = [
            (
                "natural_language_default",
                "Default to natural language. Output code only when requested or clearly implied.",
            ),
            (
                "rephrase_no_code",
                "If the user asks to rephrase / different words: output exactly the requested number of rephrasings, no code.",
            ),
            (
                "correction_no_topic_jump",
                "When corrected (CORRECT/NO/etc): acknowledge, restate intent, and fix behavior. Do not invent a new task.",
            ),
            (
                "clarify_if_unsure",
                "If unsure about the request: ask 1–2 short clarifying questions, then proceed.",
            ),
        ]
        for title, desc in defaults:
            self.add(title=title, description=desc, enabled=True)
        self._save()