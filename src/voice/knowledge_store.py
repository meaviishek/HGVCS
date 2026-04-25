"""
KnowledgeStore  –  Teach V anything, persist it forever.

Data is saved to  data/v_knowledge.json  (relative to project root).
Each entry:  { "id": str, "title": str, "text": str, "added": ISO-timestamp }

Search uses TF-IDF-style keyword overlap – no external ML needed.
"""

import json
import logging
import os
import re
import time
import uuid
from typing import List, Dict, Optional, Tuple

log = logging.getLogger("hgvcs.knowledge")

# ── storage path ───────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "..", "..", "data")
_KNOWLEDGE_FILE = os.path.join(_DATA_DIR, "v_knowledge.json")

# ── stop-words (excluded from scoring) ─────────────────────
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "about", "from", "into", "through", "and",
    "or", "but", "if", "then", "that", "this", "these", "those", "it",
    "its", "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "they", "them", "their", "what", "which", "who", "how", "when",
    "where", "why", "not", "no", "so", "just", "very", "also",
}


def _tokenize(text: str) -> List[str]:
    """Lower-case word tokens, no punctuation, no stop-words."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _score(query_tokens: List[str], entry_tokens: List[str]) -> float:
    """Jaccard-like overlap score between query and entry token sets."""
    if not query_tokens or not entry_tokens:
        return 0.0
    q = set(query_tokens)
    e = set(entry_tokens)
    intersection = q & e
    if not intersection:
        return 0.0
    return len(intersection) / (len(q | e) + 1e-9)


class KnowledgeStore:
    """
    Persistent knowledge base for V.

    Usage:
        ks = KnowledgeStore()
        ks.add_text("Paris", "Paris is the capital of France.")
        snippets = ks.search("what is the capital of France", top_k=3)
    """

    def __init__(self, path: str = _KNOWLEDGE_FILE):
        self._path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._items: List[Dict] = self._load()
        # pre-tokenize for fast search
        self._tokens: Dict[str, List[str]] = {
            item["id"]: _tokenize(item["title"] + " " + item["text"])
            for item in self._items
        }
        log.info(f"KnowledgeStore loaded {len(self._items)} items from {path}")

    # ── public API ─────────────────────────────────────────

    def add_text(self, title: str, text: str) -> str:
        """Add a text snippet. Returns the new entry id."""
        if not text.strip():
            raise ValueError("Text cannot be empty")
        entry = {
            "id":    str(uuid.uuid4())[:8],
            "title": title.strip() or "Untitled",
            "text":  text.strip(),
            "added": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._items.append(entry)
        self._tokens[entry["id"]] = _tokenize(entry["title"] + " " + entry["text"])
        self._save()
        log.info(f"Knowledge added: '{entry['title']}' ({len(text)} chars)")
        return entry["id"]

    def add_file(self, path: str) -> str:
        """
        Extract text from a file and add it.
        Supports: .txt .md .csv .pdf .docx
        Returns the new entry id.
        """
        ext = os.path.splitext(path)[1].lower()
        title = os.path.basename(path)

        if ext in (".txt", ".md", ".csv"):
            text = self._read_plain(path)
        elif ext == ".pdf":
            text = self._read_pdf(path)
        elif ext in (".docx", ".doc"):
            text = self._read_docx(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        if not text.strip():
            raise ValueError("File appears to be empty or unreadable")

        return self.add_text(title, text)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Return the top-k most relevant knowledge items for *query*.
        Each result: {"title": str, "text": str, "score": float}
        """
        q_tokens = _tokenize(query)
        scored: List[Tuple[float, Dict]] = []
        for item in self._items:
            e_tokens = self._tokens.get(item["id"], [])
            s = _score(q_tokens, e_tokens)
            if s > 0:
                scored.append((s, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"title": it["title"], "text": it["text"], "score": sc}
            for sc, it in scored[:top_k]
        ]

    def build_context(self, query: str, top_k: int = 3,
                      max_chars: int = 800) -> str:
        """
        Build a context string to prepend to the LLM prompt.
        Returns empty string if nothing relevant is found.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return ""
        chunks = []
        total = 0
        for r in results:
            snippet = f"[{r['title']}] {r['text']}"
            if total + len(snippet) > max_chars:
                snippet = snippet[: max_chars - total]
            chunks.append(snippet)
            total += len(snippet)
            if total >= max_chars:
                break
        context = "\n".join(chunks)
        return f"RELEVANT KNOWLEDGE:\n{context}\n"

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by id. Returns True if found."""
        before = len(self._items)
        self._items = [i for i in self._items if i["id"] != entry_id]
        self._tokens.pop(entry_id, None)
        if len(self._items) < before:
            self._save()
            return True
        return False

    def clear(self):
        """Remove all knowledge entries."""
        self._items = []
        self._tokens = {}
        self._save()
        log.info("KnowledgeStore cleared")

    def all_items(self) -> List[Dict]:
        """Return a copy of all entries (newest first)."""
        return list(reversed(self._items))

    def count(self) -> int:
        return len(self._items)

    # ── persistence ────────────────────────────────────────

    def _load(self) -> List[Dict]:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Could not load knowledge file: {e}")
        return []

    def _save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._items, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning(f"Could not save knowledge file: {e}")

    # ── file readers ───────────────────────────────────────

    @staticmethod
    def _read_plain(path: str) -> str:
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        return ""

    @staticmethod
    def _read_pdf(path: str) -> str:
        # Try pypdf first, then pdfplumber
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except ImportError:
            pass
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(
                    p.extract_text() or "" for p in pdf.pages
                )
        except ImportError:
            log.warning("PDF support needs: pip install pypdf  or  pip install pdfplumber")
            return ""

    @staticmethod
    def _read_docx(path: str) -> str:
        try:
            import docx
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            log.warning("DOCX support needs: pip install python-docx")
            return ""
