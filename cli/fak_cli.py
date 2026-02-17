# fak_core/cli/fak_cli.py

from runtime.runtime import Runtime


def main() -> None:
    rt = Runtime()
    print("FAK Runtime CLI. Type ':help' for commands.")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        # -------------------------
        # Colon-prefixed commands
        # -------------------------
        if line.startswith(":"):
            cmd, *rest = line[1:].split(None, 1)
            arg = rest[0] if rest else ""

            if cmd in ("q", "quit", "exit"):
                print("Bye.")
                break

            if cmd == "help":
                print(
                    "Commands:\n"
                    "  :help                      - show this help\n"
                    "  :q / :quit / :exit         - exit\n\n"
                    "Design flow:\n"
                    "  :new                       - clear current design; next input starts a new one\n"
                    "  :load <design_id>          - load a design by id\n"
                    "  :freeze [label]            - freeze current design into skills + patterns\n"
                    "                              optional label, e.g. :freeze tone_player_v1\n\n"
                    "  :clear                     - clear chat history\n"
                    "Memory:\n"
                    "  :skills                    - list known skill ids\n"
                    "  :skill_show <skill_id>     - show stored core_code for a skill\n"
                    "  :patterns                  - list known reasoning pattern ids\n"
                    "  :pattern_show <pattern_id> - show template for a pattern\n"
                    "  :mpm [n]                   - show last n MPM moments (default 3)\n"
                    "  :embryo                    - show embryo self-state snapshot\n\n"
                    "Contracts (soft workflow rules):\n"
                    "  :contracts                 - list contracts\n"
                    "  :contract_show <id>         - show one contract\n"
                    "  :contract_on <id>           - enable a contract\n"
                    "  :contract_off <id>          - disable a contract\n"
                    "  :contract_add <title>|<desc> - add a contract\n"
                    "                                stored/printed format: \"title\" description\n"
                    "  :contract_reset            - reset contracts to defaults\n\n"
                    "Utilities:\n"
                    "  :offline                   - process one offline consolidation job\n"
                    "  :deep                      - print a DEEP_REPAIR prompt template\n"
                    "  :rethink                   - print a RETHINK_PATH prompt template\n"
                    "  :vision                    - print a FAK/OMFE status-check prompt template\n"
                )
                continue

            # ---------------- Contracts ----------------

            if cmd == "contracts":
                ids = rt.contracts.all_ids()
                if not ids:
                    print("No contracts stored yet.")
                else:
                    print("Contracts:")
                    for cid in ids:
                        c = rt.contracts.get(cid)
                        if not c:
                            continue
                        status = "ON" if c.enabled else "OFF"
                        # required format: "title" description
                        print(f"  - {cid} [{status}] \"{c.title}\" {c.description}")
                continue

            if cmd == "contract_show":
                cid = arg.strip()
                if not cid:
                    print("Usage: :contract_show <contract_id>")
                    continue
                c = rt.contracts.get(cid)
                if not c:
                    print(f"No contract found with id '{cid}'.")
                else:
                    print(f"Contract: {c.id} [{'ON' if c.enabled else 'OFF'}]")
                    print(f"\"{c.title}\" {c.description}")
                continue

            if cmd in ("contract_on", "contract_off"):
                cid = arg.strip()
                if not cid:
                    print(f"Usage: :{cmd} <contract_id>")
                    continue
                ok = rt.contracts.set_enabled(cid, enabled=(cmd == "contract_on"))
                print("OK" if ok else f"No contract '{cid}'.")
                continue

            if cmd == "contract_add":
                if not arg or "|" not in arg:
                    print("Usage: :contract_add <title>|<description>")
                    continue
                title, desc = arg.split("|", 1)
                c = rt.contracts.add(title=title.strip(), description=desc.strip())
                print(f"Added contract {c.id} [ON]: \"{c.title}\" {c.description}")
                continue

            if cmd == "contract_reset":
                rt.contracts.reset_defaults()
                print("Contracts reset to defaults.")
                continue

            # ---------------- Design flow ----------------
            if cmd == "clear":
                try:
                    from runtime.model_client import AnythingLLMBackend
                    backend = AnythingLLMBackend()
                    ok = backend.clear_chat_history()
                    print("Chat history cleared!" if ok else "Failed to clear (do it manually in UI)")
                except Exception as e:
                    print(f"Error: {e}")
                continue

            if cmd == "freeze":
                label = arg.strip() or None
                resp = rt.freeze_current_design(label=label)
                print(resp)
                print(rt.reset_design())
                continue

            if cmd == "new":
                print(rt.reset_design())
                continue

            if cmd == "load":
                if not arg:
                    print("Usage: :load <design_id>")
                else:
                    try:
                        design = rt.load_design(arg.strip())
                        print(f"Loaded design {design.id} | goal: {design.goal}")
                    except Exception as e:
                        print(f"Failed to load design: {e}")
                continue

            # ---------------- Memory ----------------

            if cmd == "skills":
                ids = rt.skills.all_ids()
                if not ids:
                    print("No skills stored yet.")
                else:
                    print("Skills:")
                    for sid in ids:
                        print(f"  - {sid}")
                continue

            if cmd == "skill_show":
                sid = arg.strip()
                if not sid:
                    print("Usage: :skill_show <skill_id>")
                    continue
                card = rt.skills.get_skill(sid)
                if card is None:
                    print(f"No skill found with id '{sid}'.")
                else:
                    print(f"Skill: {card.id}")
                    if getattr(card, "summary", ""):
                        print(f"Summary: {card.summary}")
                    print("Core code:\n")
                    print(card.core_code or "(no core_code stored)")
                continue

            if cmd == "patterns":
                ids = rt.patterns.all_ids()
                if not ids:
                    print("No patterns stored yet.")
                else:
                    print("Patterns:")
                    for pid in ids:
                        print(f"  - {pid}")
                continue

            if cmd == "pattern_show":
                pid = arg.strip()
                if not pid:
                    print("Usage: :pattern_show <pattern_id>")
                    continue
                patt = rt.patterns.get_pattern(pid)
                if patt is None:
                    print(f"No pattern found with id '{pid}'.")
                else:
                    print(f"Pattern: {patt.id}")
                    if getattr(patt, "summary", ""):
                        print(f"Summary: {patt.summary}")
                    print("Template:\n")
                    print(patt.template or "(no template stored)")
                continue

            if cmd == "embryo":
                print(rt.embryo_snapshot())
                continue

            if cmd == "mpm":
                n = 3
                if arg.strip():
                    try:
                        n = int(arg.strip())
                    except ValueError:
                        print("Usage: :mpm [n]")
                        continue
                print(rt.mpm_snapshot(n=n))
                continue

            # ---------------- Utilities ----------------

            if cmd == "offline":
                print(rt.tick_offline(background=False))
                continue

            if cmd == "deep":
                print(
                    "\n--- DEEP REPAIR TEMPLATE ---\n"
                    "DEEP_REPAIR:\n"
                    "Context: we already have <existing behavior> that works.\n"
                    "Problems: <list failures / confusions that happened>.\n"
                    "Change type: ADDITIVE – keep what works, fix the rest.\n"
                    "New requirement: <what we now want instead>.\n"
                    "Output: full updated script in Python (single file), minimal and runnable.\n"
                )
                continue

            if cmd == "rethink":
                print(
                    "\n--- RETHINK PATH TEMPLATE ---\n"
                    "RETHINK_PATH:\n"
                    "Short goal: <1–2 lines about what this should do>.\n"
                    "What we already have: <quick description of current behavior>.\n"
                    "What's going wrong: <1–3 bullets>.\n"
                    "Request: restate goal in 2 lines, then give a clean next iteration.\n"
                )
                continue

            if cmd == "vision":
                print(
                    "\n--- FAK / OMFE STATUS CHECK ---\n"
                    "Based on everything so far, state:\n"
                    "- Our core FAK/OMFE vision\n"
                    "- Today’s sub-goal\n"
                    "- The next 2–3 concrete actions you propose\n"
                )
                continue

            print(f"Unknown command: :{cmd}")
            continue

        # -------------------------
        # Normal user input
        # -------------------------
        out = rt.handle_user_input(line)
        print(out)


if __name__ == "__main__":
    main()