from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import requests
import time

import os

import json
from pathlib import Path

API_BASE_URL = "http://localhost:3001/api/v1"
API_KEY = "NFRV9JW-KBZ4NVZ-KHXR6QD-FP42MBR"
WORKSPACE_SLUG = "mpm"


# ==============================
# AnythingLLM backend wrapper
# ==============================

@dataclass
class AnythingLLMBackend:
    api_key: str = API_KEY
    workspace_slug: str = WORKSPACE_SLUG
    base_url: str = API_BASE_URL

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def clear_chat_history(self) -> bool:
        """Clear the workspace chat history to prevent context pollution."""
        url = f"{self.base_url}/workspace/{self.workspace_slug}/chats"
        try:
            resp = requests.delete(url, headers=self._headers(), timeout=10)
            return resp.status_code < 400
        except Exception:
            return False

    def chat(self, message: str, mode: str = "chat") -> Dict[str, Any]:
        """
        Low-level call to AnythingLLM /workspace/{slug}/chat.
        We keep a tight timeout so a single request can't hang forever.
        """
        url = f"{self.base_url}/workspace/{self.workspace_slug}/chat"
        payload: Dict[str, Any] = {
            "message": message,
            "mode": mode,  # "chat" or "query"
        }

        try:
            resp = requests.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=20,  # seconds; small "thinking units"
            )

            # Try to parse JSON; if it fails, keep raw text
            try:
                data = resp.json()
            except Exception:
                data = {"raw": resp.text}

            if resp.status_code >= 400:
                msg = data.get("error") if isinstance(data, dict) else None
                raise RuntimeError(
                    f"AnythingLLM HTTP {resp.status_code} | {msg or resp.text}"
                )

            return data

        except Exception as e:
            # Consistent error channel
            return {
                "textResponse": f"[AnythingLLM error] {e}",
            }

    def simple_reply(self, message: str, mode: str = "chat") -> str:
        """
        Convenience helper – just return plain text.
        """
        data = self.chat(message, mode=mode)
        # AnythingLLM uses "textResponse" for the main answer.
        text = data.get("textResponse")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return str(data)


# ==============================
# ModelClient used by Runtime
# ==============================

class ModelClient:
    """
    Thin adapter between our Runtime and AnythingLLM.
    Integrates resonance feedback, style knobs, and embryo weights
    to modulate model behavior.
    """

    def __init__(self, backend: Optional[AnythingLLMBackend] = None) -> None:
        self.backend = backend or AnythingLLMBackend()

    # ------------------------------------------------------------------ #
    # ONLINE_SOLVER – "Language of the Heart" protocol with routing
    # ------------------------------------------------------------------ #

    def online_solve(
        self,
        prompt: str,
        context: Optional[str] = None,
        resonance_state: Optional[float] = None,
        style_knobs: Optional[Dict[str, float]] = None,
        embryo_weights: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        ONLINE_SOLVER role: co-pilot during active work.
        """
        system = (
            "You are a helpful AI assistant in a local-first, single-human runtime.\n"
            "You are collaborating with ONE human designer to build tools and apps.\n\n"
            
            "RULES:\n"
            "- Respond naturally in plain language.\n"
            "- When asked for code, provide minimal, runnable scripts.\n"
            "- Do NOT output JSON unless explicitly asked.\n"
            "- Do NOT wrap responses in ```json blocks unless asked.\n"
            "- Keep answers concise and direct.\n\n"
            
            "INTERNAL REASONING (optional):\n"
            "- You MAY use <think>...</think> blocks for complex problems.\n"
            "- Inside <think>, reason step-by-step before answering.\n"
            "- The human will NOT see <think> content — only your final answer.\n\n"
            
            "MEMORY (if KNOWN_SKILLS or KNOWN_PATTERNS appear in context):\n"
            "- These are previously learned solutions and thinking styles.\n"
            "- When relevant, REUSE them instead of re-deriving from scratch.\n"
            "- Example: if you see 'text_rephrasing' skill, use that approach.\n"
        )

        # Build routing/state context from resonance, style, and embryo
        routing_context = self._build_routing_context(
            resonance_state=resonance_state,
            style_knobs=style_knobs,
            embryo_weights=embryo_weights,
        )

        if context:
            message = (
                f"[SYSTEM]\n{system}\n\n"
                f"[ROUTING_STATE]\n{routing_context}\n\n"
                f"[CONTEXT]\n{context}\n\n"
                f"[USER]\n{prompt}"
            )
        else:
            message = (
                f"[SYSTEM]\n{system}\n\n"
                f"[ROUTING_STATE]\n{routing_context}\n\n"
                f"[USER]\n{prompt}"
            )

        t0 = time.time()
        reply = self.backend.simple_reply(message, mode="chat")
        elapsed_ms = (time.time() - t0) * 1000.0
        print(f"[METRIC] online_solve completed in {elapsed_ms:.1f} ms")
        return reply


    def _build_routing_context(
        self,
        resonance_state: Optional[float] = None,
        style_knobs: Optional[Dict[str, float]] = None,
        embryo_weights: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Build a text description of current system state for the model.
        This modulates the model's behavior without changing the core system prompt.
        Called by online_solve() to inject routing hints.
        """
        lines = []

        if resonance_state is not None:
            # Resonance: 0=struggling, 0.5=neutral, 1.0=thriving
            if resonance_state < 0.3:
                resonance_note = (
                    "The system is STRUGGLING (recent responses were not useful). "
                    "You should provide extra clarity, ask clarifying questions, "
                    "and avoid assumptions."
                )
            elif resonance_state < 0.7:
                resonance_note = (
                    "The system is NEUTRAL. Continue standard reasoning, "
                    "but be ready to adapt if the human corrects you."
                )
            else:
                resonance_note = (
                    "The system is THRIVING (recent responses were useful). "
                    "You may be slightly more confident and offer higher-level insights."
                )
            lines.append(f"[RESONANCE] {resonance_note}")
            lines.append(f"[RESONANCE_VALUE] {resonance_state:.2f}")

        if style_knobs:
            reflection = style_knobs.get("reflection_allowed", 0.5)
            verbosity = style_knobs.get("verbosity", 0.5)
            directness = style_knobs.get("directness", 0.5)

            if reflection > 0.7:
                lines.append("[STYLE] reflection_allowed: YES—include <think> blocks and reasoning.")
            else:
                lines.append("[STYLE] reflection_allowed: NO—keep <think> blocks minimal.")

            if verbosity > 0.7:
                lines.append("[STYLE] verbosity: HIGH—provide detailed explanations.")
            elif verbosity < 0.3:
                lines.append("[STYLE] verbosity: LOW—be concise and direct.")
            else:
                lines.append("[STYLE] verbosity: MEDIUM—balanced detail and brevity.")

            if directness > 0.7:
                lines.append("[STYLE] directness: HIGH—answer directly without preamble.")
            elif directness < 0.3:
                lines.append("[STYLE] directness: LOW—be thoughtful and exploratory.")
            else:
                lines.append("[STYLE] directness: MEDIUM—direct but with context.")

        if embryo_weights:
            directness = embryo_weights.get("directness", 0.5)
            curiosity = embryo_weights.get("curiosity", 0.5)
            concreteness = embryo_weights.get("concreteness", 0.5)
            abstraction = embryo_weights.get("abstraction_tolerance", 0.5)

            lines.append(
                f"[EMBRYO] Self-tune: directness={directness:.2f}, "
                f"curiosity={curiosity:.2f}, concreteness={concreteness:.2f}, "
                f"abstraction_tolerance={abstraction:.2f}"
            )

        return "\n".join(lines) if lines else "[ROUTING_STATE] default"

    # ------------------------------------------------------------------ #
    # OFFLINE_SOLVER – strict JSON for freeze/consolidation
    # ------------------------------------------------------------------ #

    def offline_consolidate(self, instructions: str, graph_snapshot: str) -> str:
        """
        OFFLINE_SOLVER role: used during background consolidation (freeze).

        It MUST return raw JSON only, with no markdown fences or explanations.
        If it cannot find any useful skills/patterns, it MUST return:
          {"skills": [], "patterns": []}
        """
        system = (
            "You are the OFFLINE_SOLVER module for a local agent engine.\n"
            "Your ONLY job is to analyze the provided instructions / transcript\n"
            "and return a SINGLE JSON object describing skills and patterns.\n\n"
            "IMPORTANT:\n"
            "- Do NOT wrap the JSON in markdown or code fences.\n"
            "- Do NOT add commentary, explanations, or natural language.\n"
            "- If you cannot find any useful skills or patterns, respond with:\n"
            '  {\"skills\": [], \"patterns\": []}\n'
            "- Your entire output must be valid JSON that json.loads() can parse.\n"
        )

        message = (
            f"[SYSTEM]\n{system}\n\n"
            f"[INSTRUCTIONS]\n{instructions}\n\n"
            f"[GRAPH SNAPSHOT]\n{graph_snapshot}\n"
        )

        return self.backend.simple_reply(message, mode="chat")


# ---------------------------------------------------------------------- #
# MPM helper – flatten multi-perspective memory into a rich embedding
# ---------------------------------------------------------------------- #

def flatten_mpm(memory_stack: list) -> Dict[str, Any]:
    """
    Convert a multi-perspective bundle (List[MemoryMoment]) into a single
    rich nested dict, ready to be serialized or turned into an embedding.

    We don't depend directly on MemoryMoment here to avoid import cycles;
    we just expect each item to have an .as_dict() method and a
    .perspective_depth attribute.
    """
    combined: Dict[str, Any] = {}
    for m in memory_stack:
        # defensive: support both MemoryMoment and raw dicts
        if hasattr(m, "as_dict"):
            level_dict = m.as_dict()
            depth = getattr(m, "perspective_depth", level_dict.get("perspective_depth", 0))
        else:
            level_dict = dict(m)
            depth = level_dict.get("perspective_depth", 0)

        combined[f"level_{depth}"] = level_dict

    return combined
