from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import json
import uuid

from config import DATA_DIR, MEMORY_DIR

# New location (inside memory/)
DOCS_FILE = MEMORY_DIR / "docs_store.json"
# Old location (older builds)
OLD_DOCS_FILE = DATA_DIR / "docs_store.json"


@dataclass
class DocChunk:
    id: str
    doc_id: str
    index: int
    text: str


@dataclass
class Document:
    id: str
    source: str
    kind: str
    chunks: List[DocChunk]
    meta: Dict[str, Any]


class DocsStore:
    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

        # Migrate old docs_store.json if present
        if (not DOCS_FILE.exists()) and OLD_DOCS_FILE.exists():
            try:
                DOCS_FILE.write_text(OLD_DOCS_FILE.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass

        self.documents: Dict[str, Document] = {}

        if DOCS_FILE.exists():
            try:
                raw = json.loads(DOCS_FILE.read_text(encoding="utf-8") or "{}")
            except Exception:
                raw = {}

            docs = raw.get("documents", [])
            for d in docs:
                try:
                    chunks = []
                    for c in d.get("chunks", []):
                        chunks.append(
                            DocChunk(
                                id=c["id"],
                                doc_id=c["doc_id"],
                                index=int(c["index"]),
                                text=c.get("text", ""),
                            )
                        )
                    self.documents[d["id"]] = Document(
                        id=d["id"],
                        source=d.get("source", ""),
                        kind=d.get("kind", ""),
                        chunks=chunks,
                        meta=d.get("meta", {}) or {},
                    )
                except Exception:
                    continue

    def _save(self) -> None:
        payload = {
            "documents": [
                {
                    "id": doc.id,
                    "source": doc.source,
                    "kind": doc.kind,
                    "meta": doc.meta,
                    "chunks": [
                        {"id": c.id, "doc_id": c.doc_id, "index": c.index, "text": c.text}
                        for c in doc.chunks
                    ],
                }
                for doc in self.documents.values()
            ]
        }
        DOCS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def add_document(self, source: str, kind: str, raw_text: str, meta: Dict[str, Any] | None = None) -> Document:
        doc_id = str(uuid.uuid4())[:8]
        meta = meta or {}

        # Chunking: simple fixed size
        chunk_size = 900
        chunks: List[DocChunk] = []
        text = (raw_text or "").strip()

        if text:
            parts = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        else:
            parts = []

        for idx, part in enumerate(parts):
            chunks.append(DocChunk(id=str(uuid.uuid4())[:8], doc_id=doc_id, index=idx, text=part))

        doc = Document(id=doc_id, source=source, kind=kind, chunks=chunks, meta=meta)
        self.documents[doc_id] = doc
        self._save()
        return doc

    def search_chunks(self, query: str, top_k: int = 3) -> List[DocChunk]:
        """
        Very simple keyword overlap score. (Later: embeddings / BM25.)
        """
        q = (query or "").lower().strip()
        if not q:
            return []

        q_terms = set([t for t in q.split() if len(t) >= 3])
        if not q_terms:
            return []

        scored: List[tuple[float, DocChunk]] = []
        for doc in self.documents.values():
            for c in doc.chunks:
                text_l = (c.text or "").lower()
                hits = sum(1 for t in q_terms if t in text_l)
                if hits:
                    # normalize a bit by length
                    score = hits / max(200.0, float(len(c.text)))
                    scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[: max(1, int(top_k))]]