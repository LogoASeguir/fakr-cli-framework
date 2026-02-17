# user_profile.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict
import json


PROFILE_PATH = Path("user_profile.json")


@dataclass
class UserProfile:
    """
    Very small, editable user profile that the runtime can expose
    to the model as structured context.
    """
    nickname: str = "human"
    native_language: str = "en"
    preferred_style: str = "casual"
    notes: str = ""

    @classmethod
    def load(cls) -> "UserProfile":
        if not PROFILE_PATH.exists():
            return cls()
        data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        return cls(**data)

    def save(self) -> None:
        PROFILE_PATH.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
