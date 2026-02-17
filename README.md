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
- **Runs entirely locally** with AnythingLLM (multi-agent in the future with Ollama integration)

Think of it as **session memory for AI pair programming** — instead of starting from scratch every time, FAKR remembers what worked before.

# Installation

```bash
# 1. Clone the repo
git clone https://github.com/LogoASeguir/fakr-cli-framework
cd fakr-cli-framework

# 2. Install dependencies
pip install -r requirements.txt
```

# 3. Configuration
Open **runtime/model_client.py** and edit lines 11-13 with your AnythingLLM settings:
```
API_BASE_URL = "http://localhost:3001/api/v1"
API_KEY = "your-api-key-here"
WORKSPACE_SLUG = "your-workspace-name"
```

**Run**
```bash
python main.py
```

# Basic Workflow
```
-Hello! Help me build a calculator       # Start a new design
-Instructions                            # Start the conversation
-[... work through the design ...]       # Iterate on the problem
-:freeze calculator_v1                   # Save session to memory
-:skills                                 # View learned skills
-:new                                    # Start fresh — FAKR recalls past skills
```


## CLI Commands
```
:freeze [label]	                         # Save current session to memory
:new	                                 # Start a new design session
:skills	                                 # List learned skills
:skill_show <id>	                     # View skill details
:patterns	                             # List reasoning patterns
:pattern_show <id>	                     # View pattern template
:mpm [n]	                             # View last n memory moments
:embryo	                                 # Check self-tuning state
:help	                                 # Full command list
```

## Architecture
FAKR is organized into five layers:
```
---------------------------------------------------------------
Runtime (interaction loop + clocks)
ModelClient (LLM routing layer)
Memory System (MPM, SkillStore, PatternStore, ContractStore)
Temporal Control (ClockState)
Self-Modulation Core (EmbryoCore)
---------------------------------------------------------------
Runtime (interaction loop + clocks)
    ↓
ModelClient (AnythingLLM wrapper)
    ↓
Memory (Skills / Patterns / MPM / Contracts)
    ↓
EmbryoCore (self-modulation)
```
The system is modular and designed for experimentation rather than production deployment.

## Project Status
FAKR is experimental and under active architectural refinement.
```
It is not a production system.
It is a research-oriented runtime exploring structured AI interaction patterns.
It is still under development.
```

## Roadmap
```
Multi-model backend (Ollama integration)
                    ↓
Embryo meta-learning (remember why things worked)
                    ↓
Automatic skill recall (use learned skills without prompting)
                    ↓
Structured <think> block parsing for visible reasoning
                    ↓
MPM-based long-term memory consolidation
```


## Philosophy
This project was built with the assistance of AI tools as development accelerators.
The goal was not model supremacy, but architectural exploration — understanding how structured runtime layers can augment LLM interaction in a transparent, controllable way.

## Author
Built by [Renato Pedrosa]

Part of a growing ecosystem of personal tools.





