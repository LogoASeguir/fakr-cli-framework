from __future__ import annotations
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
import PyPDF2

from .docs_store import DocsStore


def _clean_whitespace(text: str) -> str:
    return " ".join(text.split())



def ingest_url(url: str, kind: str = "url", title: Optional[str] = None) -> str:
    """
    Fetch an HTML page, extract main text, and store it as a document.

    kind can be used to tag things like "tk_docs", "python_docs", etc.
    """
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # crude but usually enough: prefer <main> or <article> if present
    main = soup.find("main") or soup.find("article") or soup.body or soup

    raw_text = _clean_whitespace(main.get_text(separator=" ", strip=True))

    store = DocsStore()
    meta = {
        "title": title or (soup.title.string.strip() if soup.title and soup.title.string else url),
        "url": url,
    }
    doc = store.add_document(source=url, kind=kind, raw_text=raw_text, meta=meta)
    return doc.id


def ingest_pdf(path: str | Path, kind: str = "pdf") -> str:
    """
    Extract text from a local PDF file and store it as chunks in DocsStore.
    Returns the created document id.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")

    text_parts: list[str] = []
    with p.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)

    raw_text = _clean_whitespace("\n".join(text_parts))

    store = DocsStore()
    meta = {"filename": p.name, "path": str(p)}
    doc = store.add_document(source=str(p), kind=kind, raw_text=raw_text, meta=meta)
    return doc.id
