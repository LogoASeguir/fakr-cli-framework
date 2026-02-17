# Fractal Knowledge System - Quick Start Guide

## Installation & Setup

### Prerequisites
```bash
# Python 3.8+
python --version

# Required packages (in your environment)
pip install numpy scipy requests

# Optional: For full tensor network features
pip install tensornetwork

# Ollama (for LLM backend)
curl https://ollama.ai/install.sh | sh
```

### Start Ollama Server
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull models (one-time)
ollama pull phi3-mini:3.8b    # 3B: fast path (~2GB)
ollama pull neural-chat:5b    # 5B: reasoning (~4GB)
ollama pull llama2:8b         # 8B: synthesis (~6GB)
```

---

## Basic Usage

### 1. Simple Query
```python
from memory.fractal_orchestration import FractalKnowledgeBase

kb = FractalKnowledgeBase()

# Ask a question
response = kb.execute_query("What is recursion?")
print(response["response"])
```

### 2. Semantic Understanding
```python
# See how query is parsed
understanding = kb.understand_query("create tkinter window")
print(f"Coordinates: {understanding['coordinates']}")
print(f"Complexity: {understanding['complexity_score']}")
print(f"Model: {understanding['recommended_model']}")
```

### 3. Knowledge Storage
```python
# Store a skill
kb.store_skill(
    id="cli_calc",
    name="Python CLI Calculator",
    code="def calc(a, op, b): return eval(f'{a}{op}{b}')",
    keywords=["python", "cli", "calculator"]
)

# Later retrieval uses this knowledge as context
response = kb.execute_query("How do I make a CLI calculator?")
```

### 4. Learning from Files
```python
from pathlib import Path

# OFFLINE_PROTOCOL: Ingest knowledge
result = kb.ingest_knowledge_from_file(Path("my_documentation.txt"))
print(f"Extracted {result['concepts_extracted']} concepts")

# These concepts now enhance all future queries
```

### 5. Concept Binding
```python
# Bind related concepts into composite
binding_id = kb.bind_skill_concepts(
    "web_scraper",
    ["skill", "web", "scraping", "python"]
)

# Find similar concepts
similar = kb.query_related_concepts("scraping", top_k=5)
for concept, similarity in similar:
    print(f"{concept}: {similarity:.2%}")
```

### 6. System Status
```python
# Full system introspection
status = kb.get_system_status()
print(f"Knowledge blocks: {status['metrics']['knowledge_blocks_stored']}")
print(f"Concepts: {status['concepts']['total']}")
print(f"Cache hits: {status['metrics']['cache_hits']}")

# Architecture explanation
arch = kb.explain_system()
print(arch["3_clocks"])
```

---

## File Formats

### Work with Geometric Data Files (.gdf)
```python
from memory.geometric_format import (
    GeometricDataFile, 
    StructureTemplate, 
    RecursiveBlock,
    GeometricStore
)

# Create new storage
gdf = GeometricDataFile()

# Register template (reusable structure)
template = StructureTemplate(
    id="skill_v1",
    shape={"name": "str", "code": "str", "tags": "list[str]"},
    purpose="Python code snippet"
)
gdf.add_template(template)

# Add blocks (compressed storage)
block = RecursiveBlock(
    template_id="skill_v1",
    coordinates=(0, 1, 2),
    field_values={
        "name": "My Skill",
        "code": "...",
        "tags": ["python"]
    }
)
gdf.add_block(block)

# Save (100-1000× compression vs JSON)
gdf.save(Path("knowledge.gdf"))

# Load later
gdf2 = GeometricDataFile.load(Path("knowledge.gdf"))
```

### Work with Hypervectors
```python
from memory.hyperdimensional import (
    ConceptStore,
    HypervectorConfig,
    HDRepresentation
)

# Create store
config = HypervectorConfig(dimension=10000)
store = ConceptStore(config)

# Add concepts
store.create_concept("python")
store.create_concept("learning")
store.create_concept("tutorial")

# Bind them (creates composite concept)
binding = store.bind_concepts(
    "py_learning_tutorial",
    ["python", "learning", "tutorial"]
)

# Query similar
similar = store.query_concepts("python", top_k=3)

# Persist
store.save(Path("concepts.hvec"))
```

### Tensor Network
```python
from memory.tensor_network import TensorNetwork
import numpy as np

net = TensorNetwork()

# Add compressed tensor (automatic SVD decomposition)
skill_tensor = np.random.randn(3, 4, 5)  # 3D skill tensor
net.add_skill_tensor(
    "my_skill",
    skill_tensor,
    max_rank=8  # Tensor train rank
)

# Retrieve info
summary = net.summary()
print(f"Compression: {summary['avg_compression_ratio']:.1f}×")

# Export for storage
state = net.export_to_dict()
```

---

## Configuration

### Customizing the Fractal System
```python
from memory.fractal_orchestration import FractalKBConfig

config = FractalKBConfig(
    # Storage
    knowledge_dir=Path("/my/knowledge/store"),
    
    # Tensor network
    tensor_max_rank=16,
    tensor_truncation_threshold=1e-6,
    
    # Hyperdimensional
    hypervector_dimension=10000,
    hypervector_representation="binary",
    
    # Ollama
    ollama_base_url="http://localhost:11434",
    prefer_local_models=True,
    
    # Execution
    max_reduction_steps=1000,
    num_execution_workers=4,
    
    # Clock intervals
    fast_clock_interval=1,
    medium_clock_interval=5,
    slow_clock_interval=20,
)

kb = FractalKnowledgeBase(config)
```

---

## Query Patterns

### Pattern 1: Simple Factual Question
```
Query: "What is Python?"
Complexity: Low (~15)
Route: 3B model (fast, 10ms)
Result: Quick factual response
```

### Pattern 2: Explanation/Analysis
```
Query: "Explain how neural networks work"
Complexity: Medium (~45)
Route: 5B model (reasoning, 30ms)
Result: Multi-step explanation with examples
```

### Pattern 3: Code Generation
```
Query: "Write a function that validates email addresses"
Complexity: High (~70)
Route: 8B model (synthesis, 100ms)
Result: Full code + explanation + testing suggestions
```

### Pattern 4: Multi-Document Synthesis
```
Query: "Summarize the differences between Python and Rust"
Context: [doc1.txt, doc2.txt, doc3.txt]
Complexity: High (with OFFLINE ingestion)
Route: 8B model
Result: Comprehensive comparison with citations
```

---

## Performance Tuning

### Memory Management
```python
# For low-memory systems
config = FractalKBConfig(
    tensor_max_rank=4,              # Lower rank = more compression
    hypervector_dimension=1000,      # Smaller vectors (faster, less precise)
    num_execution_workers=1,         # Single-threaded only
)

# For high-memory systems
config = FractalKBConfig(
    tensor_max_rank=32,              # Higher rank = better precision
    hypervector_dimension=100000,    # Large vectors (slower, more precise)
    num_execution_workers=8,         # Multi-core execution
)
```

### Cache Management
```python
kb = FractalKnowledgeBase()

# Monitor cache
status = kb.get_system_status()
cache_size = status["cache_size"]
cache_hit_ratio = (
    status["metrics"]["cache_hits"] / 
    status["metrics"]["queries_processed"]
)
print(f"Cache efficiency: {cache_hit_ratio:.1%}")

# Clear cache if needed (in fractal_orchestration.py)
kb._query_cache.clear()
```

---

## Debugging & Introspection

### Trace a Query
```python
# Get detailed understanding
understanding = kb.understand_query("create window")
print("Semantic tokens:", understanding["semantic_explanation"])
print("Coordinates:", understanding["coordinates"])
print("Confidence:", understanding["coordinates"][0]["confidence"])
```

### Check Semantic Router
```python
router_metrics = kb.semantic_router.get_metrics()
print(f"Queries processed: {router_metrics['queries_processed']}")
print(f"Cache hit ratio: {router_metrics['cache_hits'] / router_metrics['queries_processed']:.1%}")
```

### Monitor Ollama Backend
```python
ollama_metrics = kb.ollama_client.get_metrics()
print(f"Avg latency: {ollama_metrics['avg_latency_ms']:.1f}ms")
print(f"Error rate: {ollama_metrics['error_rate']:.1%}")
print(f"Routing: {ollama_metrics['complexity_routing']}")
```

### Inspect Tensor Network
```python
tensor_summary = kb.tensor_network.summary()
print(f"Total blocks: {tensor_summary['total_blocks']}")
print(f"Compression: {tensor_summary['avg_compression_ratio']:.1f}×")
print(f"Blocks by level: {tensor_summary['blocks_by_level']}")
```

### System Architecture
```python
arch = kb.explain_system()
for component, desc in arch["architecture"].items():
    print(f"{component}: {desc}")
```

---

## Common Issues & Solutions

### Issue: Ollama not found
```
Error: Connection refused (http://localhost:11434)

Solution:
1. Check Ollama is running: `ollama serve`
2. In another terminal, verify: `curl http://localhost:11434/api/tags`
3. Pull models if needed: `ollama pull phi3-mini:3.8b`
```

### Issue: Low compression ratios
```
Expected: 100-1000×
Getting: < 10×?

Check:
1. Are you using .gdf format? (raw tensors won't compress well)
2. Is tensor_max_rank set appropriately? (lower = more compression)
3. Are there many unique concepts? (compression improves with repetition)

Solution: Verify .gdf files with:
gdf = GeometricDataFile.load(Path("knowledge.gdf"))
print(gdf.summary())
```

### Issue: Slow responses
```
3B should be: ~10ms
5B should be: ~30ms
8B should be: ~100ms
Getting 500ms+?

Check:
1. Ollama CPU vs GPU: `nvidia-smi` or `ollama logs`
2. Model loaded? `ollama list`
3. System memory pressure? `free -h` or `top`

Solution: Pre-load model:
ollama pull phi3-mini:3.8b
# Keep running in background
```

### Issue: Out of memory
```
Solution:
1. Reduce hypervector_dimension (10000 → 1000)
2. Reduce tensor_max_rank (16 → 4)
3. Use model offloading (load 8B only when needed)
4. Increase num_execution_workers with caution
```

---

## Integration with Your Existing Runtime

### Add to runtime.py
```python
from memory.fractal_orchestration import FractalKnowledgeBase

class Runtime:
    def __init__(self):
        # ... existing code ...
        self.fractal_kb = FractalKnowledgeBase()
    
    def handle_query(self, user_input: str):
        # Use fractal system
        result = self.fractal_kb.execute_query(user_input)
        return result["response"]
    
    def learn_offline(self, filepath: Path):
        # OFFLINE_PROTOCOL
        return self.fractal_kb.ingest_knowledge_from_file(filepath)
```

### Hook into Existing Clocks
```python
class Runtime:
    def on_fast_clock_tick(self):  # Every 1 interaction
        # Update 3B model state
        pass
    
    def on_medium_clock_tick(self):  # Every 5 interactions
        # Refresh concept bindings
        self.fractal_kb.concept_store._recalculate_similarities()
    
    def on_slow_clock_tick(self):  # Every 20 interactions
        # Deep synthesis + save knowledge to .gdf
        self.fractal_kb.save_knowledge_to_file("checkpoint.gdf")
```

---

## Next Steps

1. **Start simple**: Run basic queries, understand response paths
2. **Store knowledge**: Add skills and patterns from your domain
3. **Ingest files**: Use OFFLINE_PROTOCOL on your documentation
4. **Monitor**: Check metrics, understand what's working
5. **Scale**: Adjust tensor ranks, hypervector dimensions for your use case
6. **Integrate**: Hook into your main Runtime, leverage across application

---

## Resources

- **Architecture Deep Dive**: See [FRACTAL_KNOWLEDGE_ARCHITECTURE.md](FRACTAL_KNOWLEDGE_ARCHITECTURE.md)
- **Module Documentation**: Docstrings in each .py file
- **Tensor Train Theory**: [Oseledets & Tyrtyshnikov, 2009]
- **Vector Symbolic Architectures**: [Kleyko et al., 2022]
- **Graph Reduction**: [Lafont, 1990] on Interaction Combinators

---

**Happy building with fractal knowledge! 🧬🚀**
