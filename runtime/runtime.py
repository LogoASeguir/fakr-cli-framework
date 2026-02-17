from __future__ import annotations

from enum import Enum
import ast
import json
import re
from typing import Optional, Dict, Any

from config import DATA_DIR, DESIGNS_DIR, LOGS_DIR, MEMORY_DIR

from .design_state import DesignState
from .jobs import JobQueue, Job
from .model_client import ModelClient
from .clocks import ClockState, ClockTier
from .mpm import MemoryMoment, generate_perspective_stack
from .embryo import EmbryoCore

from memory.docs_store import DocsStore
from memory.skills import SkillStore, SkillCard
from memory.patterns import PatternStore, ReasoningPattern
from memory.contracts import ContractStore


class Mode(str, Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"


class ResonanceChannel:
    """
    Simple '2-mic' resonance tracker.
    value ~ how well this channel has been doing recently.
    beta  ~ how quickly we update.
    """

    def __init__(self, name: str, beta: float = 0.2) -> None:
        self.name = name
        self.beta = beta
        self.value: float = 0.0

    def update(self, reward: float) -> None:
        self.value = (1.0 - self.beta) * self.value + self.beta * float(reward)


class Runtime:
    """
    Core runtime:
    - ONLINE: interactive loop via ModelClient
    - OFFLINE: consolidation jobs + freeze into skills/patterns

    Memory:
    - MPM stacks
    - Skills / Patterns stores
    - Docs store
    - Contracts store (soft workflow rules)
    - Embryo (tiny self-tuning state machine)
    """

    def __init__(self) -> None:
        self._ensure_dirs()

        self.mode: Mode = Mode.ONLINE
        self.jobs = JobQueue()
        self.model = ModelClient()
        self.current_design: Optional[DesignState] = None

        # Clocks
        self.clock_state = ClockState()
        self.clock_state.register_clock(ClockTier.FAST, interval=1)
        self.clock_state.register_clock(ClockTier.MEDIUM, interval=5)
        self.clock_state.register_clock(ClockTier.SLOW, interval=20)

        # Resonance
        self.resonance: Dict[str, ResonanceChannel] = {
            "online": ResonanceChannel("online", beta=0.2),
            "offline": ResonanceChannel("offline", beta=0.2),
        }

        # Style knobs (fed back as context, not training the LLM)
        self.style: Dict[str, float] = {
            "reflection_allowed": 0.2,
            "verbosity": 0.5,
            "directness": 0.8,
        }

        self.embryo = EmbryoCore()

        # MPM log
        self.memory_log: list[list[MemoryMoment]] = []

        # Stores
        self.skills = SkillStore()
        self.patterns = PatternStore()
        self.contracts = ContractStore()

    # --------------------------- dirs ---------------------------

    def _ensure_dirs(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        DESIGNS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------- MPM ----------------------------

    def record_moment(
        self,
        sensory: Dict[str, Any],
        internal: Dict[str, Any],
        context: Dict[str, Any],
        max_depth: int = 3,
    ) -> list[MemoryMoment]:
        internal2 = dict(internal)
        internal2["mpm_recorded"] = True
        internal2["mpm_max_depth"] = int(max_depth)

        base = MemoryMoment(sensory=sensory, internal=internal2, context=context)
        stack = generate_perspective_stack(base, max_depth=max_depth)
        self.memory_log.append(stack)
        return stack

    def mpm_snapshot(self, n: int = 3) -> str:
        if not self.memory_log:
            return "No MPM moments recorded yet."

        n = max(1, min(int(n), len(self.memory_log)))
        stacks = self.memory_log[-n:]

        lines: list[str] = []
        total = len(stacks)
        for idx, stack in enumerate(stacks, start=1):
            base = stack[0]
            ctx = base.context or {}
            internal = base.internal or {}
            sensory = base.sensory or {}

            design_id = ctx.get("design_id", "?")
            step = ctx.get("clock_step", "?")
            fired = ctx.get("fired_clocks", [])
            outcome = internal.get("outcome", "OK")

            user_text = str(sensory.get("user_text", ""))
            assistant_text = str(sensory.get("assistant_response", ""))

            if len(assistant_text) > 180:
                assistant_text = assistant_text[:177] + "..."

            mode = internal.get("mode", "?")
            online_res = float(internal.get("online_resonance", 0.0) or 0.0)
            offline_res = float(internal.get("offline_resonance", 0.0) or 0.0)

            lines.append(f"MPM {idx}/{total}: design_id={design_id} step={step} fired={fired} outcome={outcome}")
            lines.append(f"  user:      {user_text!r}")
            lines.append(f"  assistant: {assistant_text!r}")
            lines.append(f"  internal:  mode={mode} online_res={online_res:.3f} offline_res={offline_res:.3f}")
            lines.append("")

        return "\n".join(lines).rstrip()

    # ---------------------- context builders --------------------

    def build_docs_context(self, query: str, max_chars: int = 1200) -> str:
        try:
            store = DocsStore()
            chunks = store.search_chunks(query, top_k=3)
        except Exception:
            return ""

        if not chunks:
            return ""

        joined = " ".join((c.text or "") for c in chunks).strip()
        if not joined:
            return ""

        if len(joined) > max_chars:
            joined = joined[: max_chars - 3] + "..."

        return f"RELEVANT_DOCS:\n{joined}"

    def build_contracts_context(self, max_chars: int = 800) -> str:
        active = self.contracts.active()
        if not active:
            return ""

        lines: list[str] = ["ACTIVE_CONTRACTS:"]
        # Required format: "title" description
        for c in active:
            lines.append(f"  - \"{c.title}\" {c.description}")

        out = "\n".join(lines).rstrip()
        if len(out) > max_chars:
            out = out[: max_chars - 3] + "..."
        return out

    # ---------------------- style / embryo ----------------------

    def update_style_from_reply(self, user_text: str, reply: str) -> None:
        if not isinstance(reply, str):
            reply = str(reply)

        r = reply.lower()
        used_reflection_heading = ("reflection pass" in r) or ("execution of next step" in r)
        too_verbose = len(reply) > 2000

        if used_reflection_heading:
            self.style["reflection_allowed"] = max(0.0, 0.9 * float(self.style["reflection_allowed"]))

        if too_verbose:
            self.style["verbosity"] = max(0.0, 0.9 * float(self.style["verbosity"]))

        # If it starts talking about other users generically, nudge directness up (single-human sandbox).
        if "other users" in r:
            self.style["directness"] = min(1.0, float(self.style["directness"]) + 0.2)

    def embryo_snapshot(self) -> str:
        lines: list[str] = []
        lines.append("Embryo snapshot:")
        lines.append(f"  weights: {self.embryo.weights}")
        if self.embryo.self_notes:
            lines.append("  recent self-notes:")
            for note in self.embryo.self_notes[-5:]:
                lines.append(f"    - {note}")
        else:
            lines.append("  (no self-notes yet)")
        return "\n".join(lines)

    # --------------------- design lifecycle ---------------------

    def start_new_design(self, goal: str) -> DesignState:
        import uuid
        design_id = str(uuid.uuid4())
        self.current_design = DesignState(id=design_id, goal=goal)
        self.current_design.save()
        return self.current_design

    def load_design(self, design_id: str) -> DesignState:
        self.current_design = DesignState.load(design_id)
        return self.current_design

    def reset_design(self) -> str:
        self.current_design = None
        return "Current design cleared. Next message will start a new design."

    # ---------------------- online interaction ------------------

    def _detect_outcome(self, user_text: str, response: str) -> str:
        t = user_text.lower().strip()
        r = (response or "").lower()

        corrected = (
            t.startswith("correct:")
            or t == "no"
            or t.startswith("no ")
            or "you didn't understand" in t
            or "i didn't ask for any code" in t
            or "no code" in t
        )
        if corrected:
            return "MISS"

        asked_rephrase = any(k in t for k in ("rephrase", "different words", "rewrite", "say this in", "3 options"))
        code_in_response = ("```" in response) or ("import " in r and ("def " in r or "class " in r))
        if asked_rephrase and code_in_response:
            return "MISS"

        return "OK"

    def handle_user_input(self, text: str) -> str:
        if self.mode != Mode.ONLINE:
            return "Runtime is in OFFLINE mode; switch to ONLINE to interact."

        tiers_to_run = self.clock_state.advance()

        # Start new design
        if self.current_design is None:
            design = self.start_new_design(goal=text)
            design.record_turn(role="user", text=text)
            design.save()
            return f"Started new design: {design.id}\nGoal: {design.goal}"

        # Record user turn
        self.current_design.record_turn(role="user", text=text)

        # Retry detection
        retry_count = 0
        t_clean = text.strip()
        for turn in self.current_design.turns:
            if turn.role == "user" and turn.text.strip() == t_clean:
                retry_count += 1

        style_hint = (
            f"Style: reflection_allowed={self.style['reflection_allowed']:.2f}, "
            f"verbosity={self.style['verbosity']:.2f}, "
            f"directness={self.style['directness']:.2f}."
        )
        embryo_self = self.embryo.describe_self()
        contracts_ctx = self.build_contracts_context()

        context = (
            f"{self.current_design.goal}\n\n"
            f"{style_hint}\n"
            f"Embryo self-model: {embryo_self}\n"
        )
        if contracts_ctx:
            context += "\n" + contracts_ctx

        if retry_count >= 2:
            context += (
                "\n\nRETRY_HINT:\n"
                "The human repeated this exact request multiple times. "
                "Ask 1–2 short clarifying questions about what went wrong, "
                "then provide a corrected answer."
            )

        docs_ctx = self.build_docs_context(f"{self.current_design.goal}\n\n{text}")
        if docs_ctx:
            context += "\n\n" + docs_ctx

        # Pass resonance state, style knobs, and embryo weights to model
        resonance_value = self.resonance["online"].value
        response = str(self.model.online_solve(
            prompt=text,
            context=context,
            resonance_state=resonance_value,
            style_knobs=self.style,
            embryo_weights=self.embryo.weights,
        ))

        # Update style + embryo
        try:
            self.update_style_from_reply(text, response)

            lower = text.lower()
            corrected = lower.startswith("correct:") or lower.startswith("no") or ("you didn't understand" in lower)
            praised = any(p in lower for p in ("nice", "thank you", "good job", "perfect"))

            self.embryo.update_from_interaction(
                user_text=text,
                reply=response,
                corrected=corrected,
                hallucinated=False,
                praised=praised,
            )
        except Exception:
            pass

        # Record assistant turn
        self.current_design.record_turn(role="assistant", text=response)
        self.current_design.save()

        reward = 1.0 if response.strip() else 0.0
        self.resonance["online"].update(reward)

        outcome = self._detect_outcome(text, response)

        # Record MPM (best-effort)
        try:
            sensory: Dict[str, Any] = {"user_text": text, "assistant_response": response}
            internal: Dict[str, Any] = {
                "mode": self.mode.value,
                "online_resonance": self.resonance["online"].value,
                "offline_resonance": self.resonance["offline"].value,
                "style": dict(self.style),
                "embryo_weights": dict(self.embryo.weights),
                "outcome": outcome,
            }
            mpm_context: Dict[str, Any] = {
                "design_id": self.current_design.id,
                "goal": self.current_design.goal,
                "clock_step": self.clock_state.step,
                "fired_clocks": [tier.value for tier in tiers_to_run],
            }
            self.record_moment(sensory=sensory, internal=internal, context=mpm_context, max_depth=3)
        except Exception:
            pass

        if ClockTier.MEDIUM in tiers_to_run:
            self._maybe_enqueue_consolidation(text, response)

        if ClockTier.SLOW in tiers_to_run:
            self.tick_offline(background=True)

        return response

    # ---------------------- offline jobs -------------------------

    def _maybe_enqueue_consolidation(self, text: str, response: str) -> None:
        if self.current_design is None:
            return
        if self.resonance["online"].value <= 0.0:
            return

        job = Job.new(
            job_type="consolidate_session",
            params={
                "design_id": self.current_design.id,
                "goal": self.current_design.goal,
                "input": text,
                "output": response,
            },
        )
        self.jobs.enqueue(job)

    def tick_offline(self, background: bool = False) -> str:
        self.mode = Mode.OFFLINE
        job = self.jobs.dequeue()

        if not job:
            self.mode = Mode.ONLINE
            return "" if background else "No consolidation jobs in queue."

        job.status = "running"
        instructions = f"Process job of type '{job.type}'.\n\nParams:\n{job.params}"

        result = self.model.offline_consolidate(
            instructions=instructions,
            graph_snapshot="(graph snapshot TBD)",
        )

        job.status = "done"
        self.mode = Mode.ONLINE

        reward = 1.0 if isinstance(result, str) and str(result).strip() else 0.0
        self.resonance["offline"].update(reward)

        return "" if background else str(result)

    # ---------------------- freeze to memory ---------------------

    def freeze_current_design(self, label: str | None = None) -> str:
        if self.current_design is None:
            return "No active design to freeze."

        # Transcript
        lines: list[str] = []
        lines.append(f"DESIGN ID: {self.current_design.id}")
        lines.append(f"GOAL: {self.current_design.goal}")
        if label:
            lines.append(f"LABEL: {label}")
        lines.append("")
        lines.append("CONVERSATION (chronological):")
        for turn in self.current_design.turns:
            lines.append(f"{turn.role.upper()}: {turn.text}")
            lines.append("")
        transcript = "\n".join(lines)

        # Contracts appended to the OFFLINE_SOLVER too
        contracts_ctx = self.build_contracts_context()

        instructions = (
            "You are the OFFLINE_SOLVER module inside the same engine.\n"
            "Return ONLY valid JSON. No markdown. No extra words.\n\n"
            "Hard constraint: core_code MUST be JSON-safe (escape newlines as \\n).\n\n"
            "Schema:\n"
            "{\n"
            '  "skills": [\n'
            "    {\n"
            '      "id": "snake_case",\n'
            '      "summary": "1-3 lines",\n'
            '      "context_keywords": ["..."],\n'
            '      "core_code": "string",\n'
            '      "notes": "optional"\n'
            "    }\n"
            "  ],\n"
            '  "patterns": [\n'
            "    {\n"
            '      "id": "snake_case",\n'
            '      "summary": "1-3 lines",\n'
            '      "template": "steps",\n'
            '      "notes": "optional"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Now analyze this transcript and output the JSON only:\n\n"
            f"{transcript}\n"
        )
        if contracts_ctx:
            instructions += "\n" + contracts_ctx + "\n"

        raw = self.model.offline_consolidate(
            instructions=instructions,
            graph_snapshot="(not used yet)",
        )
        text = str(raw).strip()

        def _strip_fences(s: str) -> str:
            s = s.strip()
            if s.startswith("```"):
                parts = s.splitlines()
                parts = parts[1:]  # drop ``` or ```json
                if parts and parts[-1].strip().startswith("```"):
                    parts = parts[:-1]
                s = "\n".join(parts).strip()
            if s.lower().startswith("json"):
                parts = s.splitlines()
                if parts and parts[0].strip().lower() in ("json", "json:"):
                    s = "\n".join(parts[1:]).strip()
                else:
                    s = s[4:].lstrip(": \n\t")
            return s.strip()

        # Helper: auto-escape code blocks for JSON
        def escape_code_blocks_for_json(s: str) -> str:
            # Escape ```blocks``` -> preserve code but make it JSON-safe
            def esc_block(match):
                code = match.group(1)
                code_escaped = (
                    code.replace("\\", "\\\\")
                        .replace('"', '\\"')
                        .replace("\n", "\\n")
                )
                return f'"{code_escaped}"'

            # Replace all ```code``` blocks with JSON-safe strings
            s = re.sub(r"```(?:[\w+]*\n)?(.*?)```", esc_block, s, flags=re.DOTALL)
            return s

        # isolate json object region
        start = text.find("{")
        end = text.rfind("}")
        json_str = text[start : end + 1] if (start != -1 and end != -1 and start < end) else text
        json_str = _strip_fences(json_str)

        data: Dict[str, Any] | None = None
        parse_error: str | None = None

        # --- Attempt parsing loop with recursive fix ---
        attempts = []

        # Attempt 1: raw
        try:
            data = json.loads(json_str)
            attempts.append("raw json.loads")
        except Exception as e1:
            parse_error = str(e1)
            attempts.append(f"raw json.loads failed: {parse_error}")

        # Attempt 2: clean tabs
        if data is None:
            try:
                cleaned = json_str.replace("\t", " ").strip()
                data = json.loads(cleaned)
                attempts.append("cleaned tabs")
            except Exception as e2:
                parse_error = str(e2)
                attempts.append(f"clean tabs failed: {parse_error}")

        # Attempt 3: escape code blocks
        if data is None:
            try:
                auto_fixed = escape_code_blocks_for_json(json_str)
                data = json.loads(auto_fixed)
                attempts.append("auto-escaped code blocks")

                # Record learned pattern for future freezes
                self.patterns.add_pattern(
                    ReasoningPattern(
                        id="code_formatting_json",
                        summary="Auto-escape code blocks in JSON during freeze",
                        template="Detect ```code``` blocks, replace newlines with \\n, escape quotes",
                        notes="Automatically learned from failed freeze"
                    )
                )
            except Exception as e3:
                parse_error = str(e3)
                attempts.append(f"auto-escape failed: {parse_error}")

        # Attempt 4: coarse wipe core_code
        if data is None:
            try:
                coarse = re.sub(
                    r'"core_code"\s*:\s*"(?:\\.|[^"\\])*"',
                    '"core_code": ""',
                    json_str,
                    flags=re.DOTALL,
                )
                data = json.loads(coarse)
                attempts.append("wiped core_code strings")
            except Exception as e4:
                parse_error = str(e4)
                attempts.append(f"wipe core_code failed: {parse_error}")

        # Attempt 5: literal_eval fallback
        if data is None:
            try:
                data = ast.literal_eval(json_str)
                attempts.append("ast.literal_eval")
            except Exception as e5:
                parse_error = str(e5)
                attempts.append(f"literal_eval failed: {parse_error}")

        if data is None:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            debug_path = LOGS_DIR / f"freeze_{self.current_design.id}.txt"
            debug_path.write_text(text, encoding="utf-8")
            return (
                "Warning: OFFLINE_SOLVER returned non-parseable JSON.\n"
                f"Saved raw output to: {debug_path}\n"
                f"Parse error: {parse_error}\n"
                f"Attempts: {attempts}"
            )

        # Store without wiping existing stores (append-only)
        skills_data = data.get("skills", []) or []
        patterns_data = data.get("patterns", []) or []

        num_skills = 0
        for s in skills_data:
            try:
                card = SkillCard(
                    id=s["id"],
                    summary=s.get("summary", ""),
                    context_keywords=s.get("context_keywords", []),
                    core_code=s.get("core_code", ""),
                    notes=s.get("notes", ""),
                )
                self.skills.add_skill(card)
                num_skills += 1
            except Exception:
                continue

        num_patterns = 0
        for p in patterns_data:
            try:
                patt = ReasoningPattern(
                    id=p["id"],
                    summary=p.get("summary", ""),
                    template=p.get("template", ""),
                    notes=p.get("notes", ""),
                )
                self.patterns.add_pattern(patt)
                num_patterns += 1
            except Exception:
                continue

        label_info = f" (label='{label}')" if label else ""
        return (
            f"Freeze complete for design {self.current_design.id}{label_info}.\n"
            f"Stored {num_skills} skill(s) and {num_patterns} pattern(s).\n"
            f"Skill IDs now: {self.skills.all_ids()}\n"
            f"Pattern IDs now: {self.patterns.all_ids()}"
        )