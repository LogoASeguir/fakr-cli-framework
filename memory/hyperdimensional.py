"""
Hyperdimensional Computing: Geometric concept binding using VSA (Vector Symbolic Architecture).

Theory:
  - Concepts as high-dimensional vectors (10,000+ dims)
  - Binding: Bundle concepts via element-wise operations
  - Unfolding: Extract original concepts from bounds
  - Hierarchical: Stack bindings for structured knowledge

Practical:
  - 100,000-dim binary vectors: O(1) composition
  - Similarity via cosine distance
  - Noise tolerance: ~10% corruption still recovers meaning
  
Perfect for:
  - Concept entanglement (skill + library + ui_element)
  - Geometric reasoning
  - Associative retrieval
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
import numpy as np
from pathlib import Path
from enum import Enum

from config import MEMORY_DIR


# ===================================================================
# Hyperdimensional Vectors
# ===================================================================

class HDRepresentation(Enum):
    """Vector space representation."""
    BINARY = "binary"        # {0, 1}
    BIPOLAR = "bipolar"      # {-1, +1}
    BINARY_SPARSE = "sparse"  # mostly zeros


@dataclass
class HypervectorConfig:
    """Configuration for hypervector operations."""
    dimension: int = 10000        # Vector dimensionality
    representation: HDRepresentation = HDRepresentation.BINARY
    sparsity: float = 0.01         # Fraction of 1s in sparse vectors
    seed: int = 42
    similarity_threshold: float = 0.8
    
    def __post_init__(self) -> None:
        if self.dimension < 1000:
            raise ValueError(f"Dimension must be >= 1000, got {self.dimension}")


class HypervectorSpace:
    """
    High-dimensional vector space for semantic operations.
    
    VSA operations:
    - Binding: compute `A ⊗ B` (holographic product)
    - Bundling: compute `A ⊕ B` (superposition)
    - Unbinding: compute `A / B` or `A * ~B` (extraction)
    """
    
    def __init__(self, config: HypervectorConfig) -> None:
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.basis_vectors: Dict[str, np.ndarray] = {}
        self.concept_cache: Dict[str, np.ndarray] = {}
    
    def generate_vector(self, label: str, cached: bool = True) -> np.ndarray:
        """
        Generate pseudo-random hypervector for a concept.
        
        Deterministic via label hash.
        """
        if cached and label in self.concept_cache:
            return self.concept_cache[label]
        
        # Seed with label hash for reproducibility
        seed_val = (hash(label) ^ self.config.seed) % (2**31)
        local_rng = np.random.RandomState(seed_val)
        
        if self.config.representation == HDRepresentation.BINARY:
            vector = local_rng.randint(0, 2, size=self.config.dimension)
        
        elif self.config.representation == HDRepresentation.BIPOLAR:
            vector = 2 * local_rng.randint(0, 2, size=self.config.dimension) - 1
        
        elif self.config.representation == HDRepresentation.BINARY_SPARSE:
            # Mostly zeros, few ones
            vector = local_rng.binomial(
                1, self.config.sparsity, size=self.config.dimension
            )
        
        else:
            raise ValueError(f"Unknown representation: {self.config.representation}")
        
        vector = vector.astype(np.uint8)
        
        if cached:
            self.concept_cache[label] = vector
        
        return vector
    
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Binding operation (holographic product):
        Combines two vectors into a superposition carrying both semantic contents.
        
        For BINARY: element-wise XOR
        For BIPOLAR: element-wise multiplication
        """
        if len(a) != len(b):
            raise ValueError("Vectors must have same dimension")
        
        if self.config.representation in (HDRepresentation.BINARY, HDRepresentation.BINARY_SPARSE):
            return (a ^ b).astype(np.uint8)
        
        elif self.config.representation == HDRepresentation.BIPOLAR:
            return (a * b).astype(np.int8)
        
        else:
            raise ValueError(f"No binding for {self.config.representation}")
    
    def unbind(self, composite: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Unbinding operation:
        Extract semantic content of first operand from composite.
        
        a = composite / b = composite ⊗ ~b (self-inverse property)
        """
        if self.config.representation == HDRepresentation.BINARY:
            # In binary, bind is self-inverse
            return (composite ^ b).astype(np.uint8)
        
        elif self.config.representation == HDRepresentation.BIPOLAR:
            # In bipolar, multiply by inverted operand
            return (composite * b).astype(np.int8)
        
        else:
            raise ValueError(f"No unbinding for {self.config.representation}")
    
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Bundling operation (superposition):
        Combine multiple vectors without losing individual information.
        
        For BINARY: majority voting after element-wise majority over input ensemble
        For BIPOLAR: sum then sign
        """
        if not vectors:
            raise ValueError("Need at least one vector")
        
        stacked = np.stack(vectors)
        
        if self.config.representation == HDRepresentation.BINARY:
            # Majority voting: 1 if > 50% of inputs are 1
            result = np.mean(stacked, axis=0) > 0.5
            return result.astype(np.uint8)
        
        elif self.config.representation == HDRepresentation.BIPOLAR:
            # Sum then take sign
            result = np.sum(stacked, axis=0)
            return np.sign(result).astype(np.int8)
        
        else:
            raise ValueError(f"No bundling for {self.config.representation}")
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute similarity (cosine distance).
        Returns 0.0-1.0 where 1.0 = identical.
        """
        # Handle binary as boolean
        if self.config.representation == HDRepresentation.BINARY:
            matches = np.sum(a == b)
            return matches / len(a)
        
        # Hamming similarity for bipolar
        else:
            return float(np.mean(a == b))
    
    def nearest_match(
        self, query: np.ndarray, candidates: Dict[str, np.ndarray]
    ) -> Tuple[Optional[str], float]:
        """
        Find most similar concept in candidate set.
        
        Returns: (concept_label, similarity_score)
        """
        if not candidates:
            return None, 0.0
        
        best_label = None
        best_similarity = -1.0
        
        for label, vector in candidates.items():
            sim = self.similarity(query, vector)
            if sim > best_similarity:
                best_similarity = sim
                best_label = label
        
        return best_label, best_similarity


# ===================================================================
# Structured Concepts
# ===================================================================

@dataclass
class ConceptVector:
    """Labeled hypervector with metadata."""
    label: str
    vector: np.ndarray
    depth: int = 0              # Nesting level
    component_labels: List[str] = field(default_factory=list)  # Sub-concepts
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def similarity_to(self, other: "ConceptVector", hspace: HypervectorSpace) -> float:
        """Similarity to another concept."""
        return hspace.similarity(self.vector, other.vector)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "vector": self.vector.tolist(),
            "depth": self.depth,
            "component_labels": self.component_labels,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptVector":
        return cls(
            label=data["label"],
            vector=np.array(data["vector"]),
            depth=data.get("depth", 0),
            component_labels=data.get("component_labels", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConceptBinding:
    """
    Composite concept created by binding multiple vectors.
    
    Example: "tkinter_gui_skill"
      - Binds: [concept_create, concept_tkinter, concept_window]
      - Result: single high-dim vector
      - Can unbind back to originals
    """
    label: str
    composite_vector: np.ndarray
    operand_labels: List[str]  # Original concepts bound together
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "composite_vector": self.composite_vector.tolist(),
            "operand_labels": self.operand_labels,
            "metadata": self.metadata,
        }


class ConceptStore:
    """
    Persistent storage for hypervectors and bindings.
    Supports efficient retrieval via similarity.
    """
    
    def __init__(self, config: HypervectorConfig) -> None:
        self.config = config
        self.hspace = HypervectorSpace(config)
        self.concepts: Dict[str, ConceptVector] = {}
        self.bindings: Dict[str, ConceptBinding] = {}
        self.data_dir = MEMORY_DIR / "hypervectors"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def create_concept(
        self, label: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ConceptVector:
        """Register new concept."""
        if label in self.concepts:
            return self.concepts[label]
        
        vector = self.hspace.generate_vector(label)
        concept = ConceptVector(
            label=label,
            vector=vector,
            metadata=metadata or {},
        )
        self.concepts[label] = concept
        return concept
    
    def bind_concepts(
        self, label: str, operand_labels: List[str], metadata: Optional[Dict[str, Any]] = None
    ) -> ConceptBinding:
        """Create binding of multiple concepts."""
        if label in self.bindings:
            return self.bindings[label]
        
        # Get operand vectors
        operands = []
        for op_label in operand_labels:
            if op_label not in self.concepts:
                self.create_concept(op_label)
            operands.append(self.concepts[op_label].vector)
        
        # Bind recursively (pairwise)
        result = operands[0]
        for operand in operands[1:]:
            result = self.hspace.bind(result, operand)
        
        binding = ConceptBinding(
            label=label,
            composite_vector=result,
            operand_labels=operand_labels,
            metadata=metadata or {},
        )
        self.bindings[label] = binding
        return binding
    
    def unbind_concept(
        self, composite_label: str, operand_label: str
    ) -> Optional[np.ndarray]:
        """Extract component from binding."""
        if composite_label not in self.bindings:
            return None
        
        if operand_label not in self.concepts:
            return None
        
        binding = self.bindings[composite_label]
        operand_vec = self.concepts[operand_label].vector
        
        return self.hspace.unbind(binding.composite_vector, operand_vec)
    
    def query_concepts(
        self, query_label: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find most similar concepts to query.
        
        Returns: [(label, similarity), ...]
        """
        if query_label not in self.concepts:
            self.create_concept(query_label)
        
        query_vec = self.concepts[query_label].vector
        
        similarities = []
        for label, concept in self.concepts.items():
            if label != query_label:  # Skip self
                sim = self.hspace.similarity(query_vec, concept.vector)
                similarities.append((label, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save(self, filepath: Path) -> None:
        """Persist to JSON."""
        data = {
            "concepts": {
                label: concept.to_dict()
                for label, concept in self.concepts.items()
            },
            "bindings": {
                label: binding.to_dict()
                for label, binding in self.bindings.items()
            },
            "config": {
                "dimension": self.config.dimension,
                "representation": self.config.representation.value,
                "sparsity": self.config.sparsity,
            }
        }
        filepath.write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, filepath: Path) -> "ConceptStore":
        """Load from JSON."""
        data = json.loads(filepath.read_text())
        
        config_data = data.get("config", {})
        config = HypervectorConfig(
            dimension=config_data.get("dimension", 10000),
            representation=HDRepresentation[config_data.get("representation", "BINARY")],
            sparsity=config_data.get("sparsity", 0.01),
        )
        
        store = cls(config)
        
        # Load concepts
        for label, concept_dict in data.get("concepts", {}).items():
            store.concepts[label] = ConceptVector.from_dict(concept_dict)
        
        # Load bindings
        for label, binding_dict in data.get("bindings", {}).items():
            binding = ConceptBinding(
                label=binding_dict["label"],
                composite_vector=np.array(binding_dict["composite_vector"]),
                operand_labels=binding_dict["operand_labels"],
                metadata=binding_dict.get("metadata", {}),
            )
            store.bindings[label] = binding
        
        return store
    
    def summary(self) -> Dict[str, Any]:
        """Statistics."""
        return {
            "concepts": len(self.concepts),
            "bindings": len(self.bindings),
            "hypervector_dimension": self.config.dimension,
            "representation": self.config.representation.value,
            "total_vector_memory_mb": (
                (len(self.concepts) + len(self.bindings))
                * self.config.dimension
                // (8 * 1024 * 1024)
            ),
        }


# ===================================================================
# Test/Example
# ===================================================================

if __name__ == "__main__":
    # Create store with smaller dimension for test
    config = HypervectorConfig(dimension=1000)
    store = ConceptStore(config)
    
    # Create atomic concepts
    print("Creating concepts...")
    store.create_concept("skill", {"type": "template"})
    store.create_concept("creating", {"type": "action"})
    store.create_concept("tkinter", {"type": "library"})
    store.create_concept("window", {"type": "ui_element"})
    
    # Bind concepts
    print("Binding concepts...")
    binding = store.bind_concepts(
        "skill_creating_tkinter_window",
        ["skill", "creating", "tkinter", "window"],
        {"description": "Skill for creating tkinter window"}
    )
    
    print(f"\nBinding label: {binding.label}")
    print(f"Operands: {binding.operand_labels}")
    
    # Query similar concepts
    print("\nFinding similar concepts to 'window'...")
    similar = store.query_concepts("window", top_k=3)
    for label, sim in similar:
        print(f"  {label}: {sim:.3f}")
    
    # Stats
    print("\nStore Summary:")
    print(json.dumps(store.summary(), indent=2))
