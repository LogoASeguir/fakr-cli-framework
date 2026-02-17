from pathlib import Path

# Project root (file location of this config)
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
DESIGNS_DIR = DATA_DIR / "designs"
MEMORY_DIR = DATA_DIR / "memory"
LOGS_DIR = DATA_DIR / "logs"

# Default paths
DEFAULT_GRAPH_PATH = MEMORY_DIR / "graph.json"
