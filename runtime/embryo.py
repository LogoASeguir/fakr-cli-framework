from __future__ import annotations

from typing import List, Dict


class EmbryoCore:
    """
    Tiny internal 'embryo' agent: explicit state, NOT an LLM.
    Tracks a few style weights and keeps short notes about what worked.
    """

    def __init__(self) -> None:
        self.weights: Dict[str, float] = {
            "directness": 0.7,
            "curiosity": 0.3,
            "concreteness": 0.5,
            "abstraction_tolerance": 0.5,
        }
        self.self_notes: List[str] = []

    def _clamp(self, x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def _inc(self, key: str, amount: float) -> None:
        self.weights[key] = self._clamp(self.weights.get(key, 0.0) + float(amount))

    def _dec(self, key: str, amount: float) -> None:
        self.weights[key] = self._clamp(self.weights.get(key, 0.0) - float(amount))

    # ----------------------- self description -----------------------

    def describe_self(self) -> str:
        d = self.weights
        traits: List[str] = []

        if d.get("directness", 0.0) > 0.7:
            traits.append("I try to be very direct.")
        elif d.get("directness", 0.0) < 0.3:
            traits.append("I tend to be cautious and indirect.")

        if d.get("curiosity", 0.0) > 0.6:
            traits.append("I ask clarifying questions when unsure.")
        elif d.get("curiosity", 0.0) < 0.3:
            traits.append("I rarely ask clarifying questions yet.")

        if d.get("concreteness", 0.0) > 0.6:
            traits.append("I prefer concrete, runnable suggestions.")
        elif d.get("concreteness", 0.0) < 0.3:
            traits.append("I drift into vague answers sometimes.")

        if d.get("abstraction_tolerance", 0.0) > 0.6:
            traits.append("I can unpack abstract ideas.")
        elif d.get("abstraction_tolerance", 0.0) < 0.3:
            traits.append("I struggle with very abstract descriptions.")

        if not traits:
            traits.append("I’m still calibrating.")

        return " ".join(traits)

    # ----------------------- learning update ------------------------

    def update_from_interaction(
        self,
        user_text: str,
        reply: str,
        corrected: bool = False,
        hallucinated: bool = False,
        praised: bool = False,
    ) -> None:
        if corrected:
            self._inc("curiosity", 0.07)
            self._inc("concreteness", 0.05)
            self._inc("directness", 0.03)
            self.self_notes.append("Corrected: ask 1–2 clarifying questions and stay on-request.")

        if hallucinated:
            self._dec("abstraction_tolerance", 0.08)
            self._inc("concreteness", 0.08)
            self.self_notes.append("Hallucination signal: stick to given context and facts only.")

        if praised:
            self._inc("directness", 0.03)
            self._inc("concreteness", 0.03)
            self.self_notes.append("Praised: keep this response style.")

        # Keep notes bounded
        if len(self.self_notes) > 30:
            self.self_notes = self.self_notes[-30:]

    def simple_reply(self, user_text: str) -> str:
        return f"Embryo state: {self.describe_self()}"