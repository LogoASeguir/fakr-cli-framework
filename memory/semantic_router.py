"""
Semantic Router: Converts natural language queries to geometric coordinates
in the fractal knowledge tree.

Uses hierarchical decomposition to map text → structure paths.
Based on sentence embeddings + recursive coordinate generation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import json
from pathlib import Path

from config import MEMORY_DIR


@dataclass
class GeometricCoordinate:
    """
    A location in the fractal knowledge tree.
    
    depth: hierarchical level (0=root, 1=category, 2=subconcept, etc.)
    indices: tuple of indices at each level (like fractal coordinates)
    confidence: how confident we are this is the right path
    semantic_hash: hash of the semantic meaning for memoization
    """
    depth: int
    indices: Tuple[int, ...]
    confidence: float
    semantic_hash: str
    
    def as_path_string(self) -> str:
        """Convert to readable path like '/0/1/2'"""
        return "/" + "/".join(str(i) for i in self.indices)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "indices": list(self.indices),
            "confidence": self.confidence,
            "semantic_hash": self.semantic_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeometricCoordinate":
        return cls(
            depth=data["depth"],
            indices=tuple(data["indices"]),
            confidence=data["confidence"],
            semantic_hash=data["semantic_hash"],
        )


@dataclass
class SemanticToken:
    """
    Decomposed semantic unit from a query.
    
    Example: "create tkinter window"
    → [SemanticToken("create", "action"),
        SemanticToken("tkinter", "library"),
        SemanticToken("window", "ui_element")]
    """
    word: str
    semantic_type: str  # "action", "library", "ui_element", "concept", etc.
    embedding_vector: Optional[List[float]] = None  # 384-dim for Sentence-Transformers


class VocabularyTrie:
    """
    Fast hierarchical lookup for semantic tokens.
    Organizes knowledge into a prefix tree structure.
    """
    
    def __init__(self) -> None:
        self.root: Dict[str, Any] = {}
        self.depth_map: Dict[str, int] = {}  # token -> depth in tree
        self.load_default_vocabulary()
    
    def load_default_vocabulary(self) -> None:
        """Initialize with common semantic categories."""
        categories = {
            "action": ["create", "build", "design", "implement", "execute", "run", "make"],
            "library": ["tkinter", "pygame", "numpy", "pandas", "torch", "tensorflow"],
            "ui_element": ["window", "button", "label", "input", "form", "widget", "panel"],
            "language": ["python", "javascript", "rust", "go", "java", "c++"],
            "pattern": ["factory", "observer", "singleton", "strategy", "adapter"],
            "concept": ["recursion", "iteration", "optimization", "scaling", "compression"],
        }
        
        depth = 0
        for category, tokens in categories.items():
            self.root[category] = {}
            for token in tokens:
                self.root[category][token] = {"leaf": True}
                self.depth_map[token] = depth
    
    def find_path(self, token: str) -> Optional[Tuple[str, ...]]:
        """Find hierarchical path for a token."""
        for category, subtree in self.root.items():
            if token in subtree:
                return (category, token)
        return None


class FractalCoordinateGenerator:
    """
    Convert semantic tokens → geometric coordinates using recursive decomposition.
    
    Key insight: Each word is a "compartment" containing:
    - Solutions (skills)
    - Patterns (reasoning)
    - Subconcepts (recursive)
    """
    
    def __init__(self) -> None:
        self.vocab_trie = VocabularyTrie()
        self.coordinate_cache: Dict[str, GeometricCoordinate] = {}
        self.structure_depth = 4  # Max recursion depth
    
    def decompose_query(self, query: str) -> List[SemanticToken]:
        """
        Split query into semantic units.
        Example: "create tkinter window" → [create, tkinter, window]
        """
        words = query.lower().strip().split()
        tokens = []
        
        for word in words:
            path = self.vocab_trie.find_path(word)
            if path:
                semantic_type = path[0]
            else:
                semantic_type = "unknown"
            
            tokens.append(SemanticToken(word=word, semantic_type=semantic_type))
        
        return tokens
    
    def generate_coordinate(
        self, query: str, context_depth: int = 0
    ) -> GeometricCoordinate:
        """
        Convert query → fractal coordinate path.
        
        Algorithm:
        1. Decompose into semantic tokens
        2. Hash the query for memoization
        3. Generate indices based on token hierarchy
        4. Return coordinate with confidence score
        """
        query_clean = query.lower().strip()
        
        # Check cache
        if query_clean in self.coordinate_cache:
            return self.coordinate_cache[query_clean]
        
        # Decompose
        tokens = self.decompose_query(query_clean)
        
        # Generate indices from token types
        indices = []
        confidence = 1.0
        
        for i, token in enumerate(tokens):
            # Map semantic type to category index
            category_map = {
                "action": 0, "library": 1, "ui_element": 2,
                "language": 3, "pattern": 4, "concept": 5, "unknown": 6
            }
            category_idx = category_map.get(token.semantic_type, 6)
            token_idx = hash(token.word) % 128  # Distribute tokens in 128 buckets
            
            # Combine indices
            combined_idx = (category_idx * 128 + token_idx) % 1024
            indices.append(combined_idx)
            
            # Confidence degrades with more tokens (ambiguity grows)
            confidence *= 0.95
        
        # Limit depth
        indices = indices[:self.structure_depth]
        
        # Create coordinate
        semantic_hash = hash(query_clean) & 0xFFFFFF  # 24-bit hash
        coord = GeometricCoordinate(
            depth=len(indices) + context_depth,
            indices=tuple(indices),
            confidence=max(0.1, confidence),
            semantic_hash=f"{semantic_hash:06x}",
        )
        
        # Cache
        self.coordinate_cache[query_clean] = coord
        return coord
    
    def compound_coordinates(
        self, coords: List[GeometricCoordinate]
    ) -> GeometricCoordinate:
        """
        Combine multiple coordinates (multi-part queries).
        Creates a hierarchical structure.
        """
        if not coords:
            raise ValueError("Need at least one coordinate")
        
        # Merge indices with weighted confidence
        all_indices = []
        total_confidence = 0.0
        
        for coord in coords:
            all_indices.extend(coord.indices)
            total_confidence += coord.confidence
        
        avg_confidence = total_confidence / len(coords)
        
        # Limit total depth
        all_indices = all_indices[:self.structure_depth]
        
        # Create compound hash
        hashes = [coord.semantic_hash for coord in coords]
        compound_hash = hash("".join(hashes)) & 0xFFFFFF
        
        return GeometricCoordinate(
            depth=len(all_indices),
            indices=tuple(all_indices),
            confidence=avg_confidence,
            semantic_hash=f"{compound_hash:06x}",
        )


class SemanticRouter:
    """
    Main router: handles NL → geometric paths and manages lookups.
    
    Caches frequent qeueries for O(1) lookup.
    Integrates with Memory stores for retrieval.
    """
    
    def __init__(self) -> None:
        self.generator = FractalCoordinateGenerator()
        self.route_cache: Dict[str, List[GeometricCoordinate]] = {}
        self.metrics = {
            "queries_processed": 0,
            "cache_hits": 0,
            "avg_confidence": 0.0,
        }
    
    def route_query(self, query: str) -> List[GeometricCoordinate]:
        """
        Main entry point: convert query to candidate paths.
        
        Returns list of coordinates (in case query has multiple interpretations).
        """
        query_clean = query.lower().strip()
        
        # Check cache
        if query_clean in self.route_cache:
            self.metrics["cache_hits"] += 1
            return self.route_cache[query_clean]
        
        # Generate coordinate
        primary_coord = self.generator.generate_coordinate(query_clean)
        
        # Generate related paths (nearby in coordinate space)
        related_coords = self._generate_nearby_coordinates(primary_coord)
        
        candidates = [primary_coord] + related_coords
        
        # Cache
        self.route_cache[query_clean] = candidates
        self.metrics["queries_processed"] += 1
        
        # Update confidence metric
        configs = [c.confidence for c in candidates]
        self.metrics["avg_confidence"] = sum(configs) / len(configs)
        
        return candidates
    
    def _generate_nearby_coordinates(
        self, coord: GeometricCoordinate, radius: int = 2
    ) -> List[GeometricCoordinate]:
        """
        Generate alternative paths (for ambiguity handling).
        Varies indices by ±radius.
        """
        nearby = []
        
        if len(coord.indices) == 0:
            return nearby
        
        # Vary last index
        base_indices = coord.indices[:-1]
        last_idx = coord.indices[-1]
        
        for delta in [-radius, radius]:
            new_last = max(0, last_idx + delta)
            new_indices = base_indices + (new_last,)
            
            nearby.append(GeometricCoordinate(
                depth=coord.depth,
                indices=new_indices,
                confidence=coord.confidence * 0.7,  # Lower confidence for variants
                semantic_hash=f"{hash(str(new_indices)) & 0xFFFFFF:06x}",
            ))
        
        return nearby
    
    def explain_route(self, query: str) -> Dict[str, Any]:
        """Debug/inspect: explain the routing decision."""
        coords = self.route_query(query)
        tokens = self.generator.decompose_query(query)
        
        return {
            "query": query,
            "tokens": [
                {
                    "word": t.word,
                    "type": t.semantic_type,
                } for t in tokens
            ],
            "primary_coordinate": coords[0].to_dict() if coords else None,
            "alternatives": [c.to_dict() for c in coords[1:]],
            "cache_hit": query.lower().strip() in self.route_cache,
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return performance metrics."""
        return self.metrics.copy()


# Test/Demo
if __name__ == "__main__":
    router = SemanticRouter()
    
    test_queries = [
        "create tkinter window",
        "python recursion algorithm",
        "numpy matrix multiplication",
        "design pattern factory",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = router.explain_route(query)
        print(json.dumps(result, indent=2))
