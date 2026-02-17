# FAKR: Fractal Agent Kernel Runtime
**A self-learning AI runtime that turns conversations into reusable knowledge.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange.svg)]()

---

## What It Does

FAKR is a local-first AI runtime that:

- **Tracks design sessions** with an LLM across multiple conversations
- **Automatically extracts skills** (code patterns) and **reasoning patterns** (thinking styles)
- **Builds a growing memory layer** that improves over time
- **Runs entirely locally** with AnythingLLM

Think of it as **session memory for AI pair programming** — instead of starting from scratch every time, FAKR remembers what worked before.

## Quick Start (Windows/Linux/macOS)
FAKR is designed for **local-first experimentation**. Once cloned, you can initialize everything quickly.

### Prerequisites
```
# Python 3.9+
python --version
```

# 1. Clone Repo
```bash
git clone https://github.com/LogoASeguir/fakr-cli-framework
cd fakr-cli-framework
```
# 2. Install dependencies
```bash
pip install -r requirements.txt
```
# 3. Install & Launch (Windows)
Double-click quick_start.bat or run
```bash
quick_start.bat
```
# 4. Install & Launch (Linux/macOS)
Make shell script executable and run
```bash
chmod +x quick_start.sh
./quick_start.sh
```
What it does:
- Installs Python dependencies from requirements.txt
- Checks if Ollama backend is running (optional)
- Starts the main FAKR runtime

Prompts you to edit model_client.py with your API keys or configure Ollama


## Overview
FAKR is an experimental local-first AI runtime designed to explore:
```
Multi-scale temporal processing (fast / medium / slow reasoning loops)
Structured memory formation from live interaction
Offline consolidation into reusable skills and reasoning patterns
Lightweight self-modulation via feedback signals
Rather than acting as a stateless chatbot, FAKR treats each session as a design process that can later be frozen into structured memory artifacts.
```
This project focuses on architecture and runtime behavior — not model training.

## CLI Commands
```
:freeze [label]	Save current session to memory
:new	Start a new design session
:skills	List learned skills
:skill_show <id>	View skill details
:patterns	List reasoning patterns
:pattern_show <id>	View pattern template
:mpm [n]	View last n memory moments
:embryo	Check self-tuning state
:help	Full command list
```

## Core Concepts
1. Multi-Clock Processing

FAKR operates with three independent temporal tiers:
```
FAST – interactive user loop
MEDIUM – background consolidation trigger
SLOW – deeper offline processing
```
This separation allows immediate interaction and reflective restructuring to coexist without interfering with each other.

2. Structured Memory (MPM + Stores)
Each interaction can generate:
```
MemoryMoment perspective stacks (MPM)
Skill cards (executable knowledge units)
Reasoning patterns (abstracted templates)
Soft workflow contracts
```
Memory is append-only and stored explicitly — not implicitly embedded inside the model.

3. Offline Consolidation
Sessions can be “frozen” into structured JSON artifacts:
```
{
  "skills": [...],
  "patterns": [...]
}
```
The runtime enforces strict JSON output and applies multiple parsing recovery layers to ensure robustness.

4. Self-Modulation Layer
The system tracks:
```
Online / offline resonance
Style parameters (verbosity, directness, reflection allowance)
A lightweight “Embryo” state machine adjusting internal weights
```
These signals influence future prompts but do not modify the underlying model.

## Architecture
FAKR is organized into five layers:
```
Runtime (interaction loop + clocks)
ModelClient (LLM routing layer)
Memory System (MPM, SkillStore, PatternStore, ContractStore)
Temporal Control (ClockState)
Self-Modulation Core (EmbryoCore)
```
The system is modular and designed for experimentation rather than production deployment.

## Project Status
FAKR is experimental and under active architectural refinement.
```
It is not a production system.
It is a research-oriented runtime exploring structured AI interaction patterns.
It is still under development.
```

## Philosophy
This project was built with the assistance of AI tools as development accelerators.
The goal was not model supremacy, but architectural exploration — understanding how structured runtime layers can augment LLM interaction in a transparent, controllable way.

FAKR is a sandbox for experimentation in:
```
Memory layering
Structured consolidation
Runtime feedback systems
Local-first AI workflows
```

## Author
Built by [Renato Pedrosa]

Part of a growing ecosystem of personal tools.





