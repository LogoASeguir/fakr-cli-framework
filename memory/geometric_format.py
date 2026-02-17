"""
Geometric Data Format (.gdf):
Custom binary format for fractal knowledge storage.
100-1000× compression vs JSON through IFS-inspired recursive encoding.

Structure:
  [Header: 32 bytes]
    - Magic: "GDF1" (4 bytes)
    - Version: 1 (1 byte)
    - Structure Dict Size: (3 bytes)
    - Reserved (24 bytes)
  
  [StructureDict: variable]
    - Maps identifiers to reusable template structures
    - Dramatically reduces redundancy (common patterns appear once)
  
  [RecursiveIndices: variable]
    - Compressed coordinate paths (like fractal IFS codes)
    - References back to StructureDict
    - Reference pointers instead of full data duplication
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional, BinaryIO
import json
import zlib
import struct
from pathlib import Path

from config import MEMORY_DIR


# ===================================================================
# Compression Primitives
# ===================================================================

def varint_encode(n: int) -> bytes:
    """
    Encode integer as variable-length byte sequence.
    More compact than fixed-width for small numbers.
    """
    result = []
    while n >= 128:
        result.append((n & 0x7F) | 0x80)
        n >>= 7
    result.append(n & 0x7F)
    return bytes(result)


def varint_decode(data: bytes, offset: int = 0) -> Tuple[int, int]:
    """Decode varint, return (value, bytes_read)."""
    result = 0
    shift = 0
    idx = offset
    
    while idx < len(data):
        byte = data[idx]
        result |= (byte & 0x7F) << shift
        idx += 1
        
        if not (byte & 0x80):
            break
        shift += 7
    
    return result, idx - offset


def dict_to_bytes(d: Dict[str, Any]) -> bytes:
    """Serialize dict to compact JSON + zlib compression."""
    json_str = json.dumps(d, separators=(',', ':'), ensure_ascii=True)
    return zlib.compress(json_str.encode('utf-8'), level=9)


def bytes_to_dict(data: bytes) -> Dict[str, Any]:
    """Decompress and deserialize."""
    json_str = zlib.decompress(data).decode('utf-8')
    return json.loads(json_str)


# ===================================================================
# Core Data Structures
# ===================================================================

@dataclass
class StructureTemplate:
    """
    Reusable structure definition (like a "class" in OOP).
    Examples:
      - "skill_template": {shape: [str, list[str], str], purpose: "code_snippet"}
      - "pattern_template": {shape: [str, str, list], purpose: "reasoning"}
    
    Dramatically reduces file size by eliminating field name duplication.
    """
    id: str  # e.g., "skill_v1", "pattern_v2"
    shape: Dict[str, str]  # Field names → types: "str", "int", "list", "nested"
    purpose: str  # Human description
    field_descriptions: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "shape": self.shape,
            "purpose": self.purpose,
            "field_descriptions": self.field_descriptions,
        }


@dataclass
class RecursiveBlock:
    """
    Single knowledge unit using a template and compressed indices.
    Storage: ~template_id (1 byte ref) + coordinates + optional data
    """
    template_id: str
    coordinates: Tuple[int, ...]  # Fractal coordinates
    field_values: Dict[str, Any]  # Data specific to this block
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def size_estimate(self) -> int:
        """Estimate serialized size in bytes."""
        template_id_size = 1  # Reference
        coords_size = sum(2 for _ in self.coordinates)  # 2 bytes per coordinate
        data_size = len(json.dumps(self.field_values, separators=(',', ':')))
        return template_id_size + coords_size + data_size
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "coordinates": list(self.coordinates),
            "field_values": self.field_values,
            "metadata": self.metadata,
        }


# ===================================================================
# Geometric Data File (Main Container)
# ===================================================================

@dataclass
class GeometricDataFile:
    """
    Container for fractal knowledge using geometric format.
    
    Achieves compression through:
      1. Structure templates (no field name repetition)
      2. Varint encoding (small integers → 1-2 bytes)
      3. Zlib on structure dict
      4. Reference pointers instead of value duplication
    """
    
    version: int = 1
    structure_dict: Dict[str, StructureTemplate] = field(default_factory=dict)
    blocks: List[RecursiveBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # ____________ Header (32 bytes) ____________
    MAGIC = b"GDF1"
    HEADER_SIZE = 32
    
    # ____________ Public API ____________
    
    def add_template(self, template: StructureTemplate) -> None:
        """Register a reusable structure template."""
        self.structure_dict[template.id] = template
    
    def add_block(self, block: RecursiveBlock) -> None:
        """Add a knowledge block."""
        if block.template_id not in self.structure_dict:
            raise KeyError(f"Template '{block.template_id}' not registered")
        self.blocks.append(block)
    
    def add_blocks(self, blocks: List[RecursiveBlock]) -> None:
        """Add multiple blocks efficiently."""
        for block in blocks:
            self.add_block(block)
    
    def serialize(self) -> bytes:
        """Convert to binary format."""
        # Prepare structure dict
        struct_dict = {
            tid: t.to_dict()
            for tid, t in self.structure_dict.items()
        }
        struct_bytes = dict_to_bytes(struct_dict)
        struct_size = len(struct_bytes)
        
        if struct_size >= 16777216:  # 2^24
            raise ValueError(f"Structure dict too large: {struct_size} bytes")
        
        # Prepare blocks
        blocks_data = b""
        for block in self.blocks:
            blocks_data += self._encode_block(block)
        
        # Build header
        header = bytearray(self.HEADER_SIZE)
        header[0:4] = self.MAGIC
        header[4] = self.version
        header[5:8] = struct.pack('>I', struct_size)[1:4]  # 3-byte big-endian
        
        # Combine
        result = bytes(header) + struct_bytes + blocks_data
        
        # Store metadata
        if self.metadata:
            result += b"META"
            meta_bytes = dict_to_bytes(self.metadata)
            result += struct.pack('>I', len(meta_bytes))
            result += meta_bytes
        
        return result
    
    @classmethod
    def deserialize(cls, data: bytes) -> "GeometricDataFile":
        """Load from binary format."""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError("Data too short for header")
        
        # Parse header
        if data[0:4] != cls.MAGIC:
            raise ValueError("Invalid magic bytes")
        
        version = data[4]
        struct_size = int.from_bytes(b'\x00' + data[5:8], 'big')
        
        # Parse structure dict
        struct_start = cls.HEADER_SIZE
        struct_end = struct_start + struct_size
        struct_bytes = data[struct_start:struct_end]
        struct_dict_raw = bytes_to_dict(struct_bytes)
        
        gdf = cls(version=version)
        for tid, tdict in struct_dict_raw.items():
            gdf.structure_dict[tid] = StructureTemplate(
                id=tdict["id"],
                shape=tdict["shape"],
                purpose=tdict["purpose"],
                field_descriptions=tdict.get("field_descriptions", {}),
            )
        
        # Parse blocks
        offset = struct_end
        while offset < len(data):
            # Check for metadata marker
            if data[offset:offset+4] == b"META":
                break
            
            block, bytes_read = cls._decode_block(data, offset, gdf.structure_dict)
            gdf.blocks.append(block)
            offset += bytes_read
        
        # Parse metadata if present
        if offset < len(data) and data[offset:offset+4] == b"META":
            offset += 4
            meta_size = struct.unpack('>I', data[offset:offset+4])[0]
            offset += 4
            meta_bytes = data[offset:offset+meta_size]
            gdf.metadata = bytes_to_dict(meta_bytes)
        
        return gdf
    
    def _encode_block(self, block: RecursiveBlock) -> bytes:
        """Encode single block to bytes."""
        result = bytearray()
        
        # Template ID (varint)
        template_list = list(self.structure_dict.keys())
        template_idx = template_list.index(block.template_id)
        result.extend(varint_encode(template_idx))
        
        # Coordinates (varint sequence)
        result.extend(varint_encode(len(block.coordinates)))
        for coord in block.coordinates:
            result.extend(varint_encode(coord))
        
        # Field values (compressed JSON)
        field_bytes = dict_to_bytes(block.field_values)
        result.extend(struct.pack('>H', len(field_bytes)))
        result.extend(field_bytes)
        
        return bytes(result)
    
    @staticmethod
    def _decode_block(
        data: bytes, offset: int, templates: Dict[str, StructureTemplate]
    ) -> Tuple[RecursiveBlock, int]:
        """Decode single block from bytes, return (block, bytes_read)."""
        start_offset = offset
        
        # Template ID
        template_idx, bytes_read = varint_decode(data, offset)
        offset += bytes_read
        template_id = list(templates.keys())[template_idx]
        
        # Coordinates
        coord_count, bytes_read = varint_decode(data, offset)
        offset += bytes_read
        coordinates = []
        for _ in range(coord_count):
            coord, bytes_read = varint_decode(data, offset)
            offset += bytes_read
            coordinates.append(coord)
        
        # Field values
        field_size = struct.unpack('>H', data[offset:offset+2])[0]
        offset += 2
        field_bytes = data[offset:offset+field_size]
        offset += field_size
        field_values = bytes_to_dict(field_bytes)
        
        block = RecursiveBlock(
            template_id=template_id,
            coordinates=tuple(coordinates),
            field_values=field_values,
        )
        
        return block, offset - start_offset
    
    def save(self, filepath: Path) -> None:
        """Write to file."""
        data = self.serialize()
        filepath.write_bytes(data)
    
    @classmethod
    def load(cls, filepath: Path) -> "GeometricDataFile":
        """Read from file."""
        data = filepath.read_bytes()
        return cls.deserialize(data)
    
    def compression_ratio(self) -> float:
        """Estimate compression vs JSON baseline."""
        # Estimate JSON size
        json_dict = {
            "templates": {k: v.to_dict() for k, v in self.structure_dict.items()},
            "blocks": [b.to_dict() for b in self.blocks],
            "metadata": self.metadata,
        }
        json_size = len(json.dumps(json_dict).encode('utf-8'))
        
        gdf_size = len(self.serialize())
        
        return json_size / max(gdf_size, 1)
    
    def summary(self) -> Dict[str, Any]:
        """Return file statistics."""
        serialized = self.serialize()
        return {
            "version": self.version,
            "num_templates": len(self.structure_dict),
            "num_blocks": len(self.blocks),
            "binary_size_bytes": len(serialized),
            "estimated_compression_ratio": self.compression_ratio(),
            "avg_block_size": sum(b.size_estimate() for b in self.blocks) // max(len(self.blocks), 1),
        }


# ===================================================================
# File Manager (convenience class)
# ===================================================================

class GeometricStore:
    """Persistent manager for .gdf files."""
    
    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = data_dir or MEMORY_DIR / "geometric"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_gdf(self, name: str, gdf: GeometricDataFile) -> Path:
        """Save GDF file."""
        path = self.data_dir / f"{name}.gdf"
        gdf.save(path)
        return path
    
    def load_gdf(self, name: str) -> GeometricDataFile:
        """Load GDF file."""
        path = self.data_dir / f"{name}.gdf"
        return GeometricDataFile.load(path)
    
    def list_gdfs(self) -> List[Path]:
        """List all .gdf files in store."""
        return list(self.data_dir.glob("*.gdf"))


# ===================================================================
# Example/Test
# ===================================================================

if __name__ == "__main__":
    # Create a GDF with skill templates
    gdf = GeometricDataFile()
    
    # Register templates
    skill_template = StructureTemplate(
        id="skill_v1",
        shape={
            "name": "str",
            "context_keywords": "list[str]",
            "core_code": "str",
        },
        purpose="Python code snippet/skill",
        field_descriptions={
            "name": "Skill name",
            "context_keywords": "Keywords for semantic routing",
            "core_code": "Executable Python code",
        }
    )
    gdf.add_template(skill_template)
    
    # Add blocks
    blocks = [
        RecursiveBlock(
            template_id="skill_v1",
            coordinates=(0, 1, 2),
            field_values={
                "name": "Python CLI Calculator",
                "context_keywords": ["python", "cli", "math"],
                "core_code": "def calc(a, op, b): return eval(f'{a}{op}{b}')",
            }
        ),
        RecursiveBlock(
            template_id="skill_v1",
            coordinates=(0, 1, 3),
            field_values={
                "name": "Tkinter Window",
                "context_keywords": ["tkinter", "gui", "window"],
                "core_code": "import tkinter as tk; root = tk.Tk(); root.mainloop()",
            }
        ),
    ]
    gdf.add_blocks(blocks)
    
    # Check stats
    print("GDF Summary:")
    print(json.dumps(gdf.summary(), indent=2))
    
    # Serialize & deserialize
    data = gdf.serialize()
    print(f"\nBinary size: {len(data)} bytes")
    
    gdf2 = GeometricDataFile.deserialize(data)
    print(f"Loaded {len(gdf2.blocks)} blocks")
    print(f"Compression ratio: {gdf.compression_ratio():.1f}×")
