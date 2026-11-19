"""OCR RAG pipeline: PyTesseract -> layout-aware chunking -> sentence-transformers -> cosine retrieval."""

import os
import re
import time

import numpy as np
from PIL import Image


# --- OCR ---

def ocr_page(image: Image.Image) -> str:
    """Run PyTesseract on a single page image. Returns raw text."""
    import pytesseract
    # Set Tesseract path if not in system PATH
    if os.name == 'nt':  # Windows
        default = r'C:\Users\320248840\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
        if os.path.exists(default):
            pytesseract.pytesseract.tesseract_cmd = default
    return pytesseract.image_to_string(image)


def ocr_pages(images: list[Image.Image],
              corpus_ids: list[str] = None) -> list[dict]:
    """OCR all page images.
    Returns list of {corpus_id, text, n_chars}.
    """
    results = []
    total = len(images)

    for i, img in enumerate(images):
        cid = corpus_ids[i] if corpus_ids else str(i)
        text = ocr_page(img)
        results.append({
            "corpus_id": cid,
            "text": text,
            "n_chars": len(text),
        })

        if (i + 1) % 50 == 0 or i + 1 == total:
            print(f"  OCR: {i+1}/{total} pages")

    empty = sum(1 for r in results if r["n_chars"] < 10)
    print(f"  {total} pages OCR'd, {empty} nearly empty")
    return results


# --- Chunking ---

def chunk_text(pages: list[dict], max_chars: int = 1500,
               overlap: int = 200) -> list[dict]:
    """Layout-aware chunking. Respects paragraph boundaries.

    Each chunk carries its source corpus_id so retrieval maps back to pages.
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"].strip()
        if len(text) < 10:
            continue

        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r"\n{2,}", text)

        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 1 > max_chars and current_chunk:
                chunks.append({
                    "chunk_id": chunk_id,
                    "corpus_id": page["corpus_id"],
                    "text": current_chunk.strip(),
                })
                chunk_id += 1
                # Overlap: keep tail of previous chunk
                if overlap > 0:
                    current_chunk = current_chunk[-overlap:] + "\n" + para
                else:
                    current_chunk = para
            else:
                current_chunk = current_chunk + "\n" + para if current_chunk else para

        if current_chunk.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "corpus_id": page["corpus_id"],
                "text": current_chunk.strip(),
            })
            chunk_id += 1

    print(f"  {len(pages)} pages -> {len(chunks)} chunks "
          f"(avg {sum(len(c['text']) for c in chunks) // max(len(chunks), 1)} chars)")
    return chunks


# --- Embedding ---

def embed_chunks(chunks: list[dict],
                 model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed text chunks with sentence-transformers (single-vector).
    Returns (n_chunks, dim) array.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"  {len(chunks)} chunks -> embeddings shape {embeddings.shape}")
    return embeddings


# --- Retrieval ---

def search_chunks(query: str, chunk_embeddings: np.ndarray,
                  chunks: list[dict], model,
                  top_k: int = 5) -> list[dict]:
    """Cosine similarity retrieval over text chunks.

    Args:
        query: query string
        chunk_embeddings: (n_chunks, dim) precomputed
        chunks: list of chunk dicts
        model: SentenceTransformer instance (for encoding query)
        top_k: number of results

    Returns:
        list of {chunk_id, corpus_id, text, score, rank}
    """
    q_emb = model.encode([query], convert_to_numpy=True)

    # Cosine similarity
    norms_c = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    norms_q = np.linalg.norm(q_emb, axis=1, keepdims=True)
    # ponytail: avoid div-by-zero with clip
    sims = (q_emb @ chunk_embeddings.T) / (norms_q * norms_c.T).clip(1e-8)
    sims = sims[0]  # (n_chunks,)

    top_k = min(top_k, len(sims))
    top_idx = np.argpartition(sims, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    results = []
    for rank, idx in enumerate(top_idx):
        results.append({
            **chunks[idx],
            "score": float(sims[idx]),
            "rank": rank,
        })
    return results


# --- Full OCR Pipeline ---

class OCRPipeline:
    """Wraps the full OCR RAG flow for evaluation integration."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.chunks = None
        self.chunk_embeddings = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index_pages(self, images: list[Image.Image],
                    corpus_ids: list[str]) -> None:
        """Run full OCR pipeline: extract -> chunk -> embed."""
        print("OCR Pipeline: indexing pages...")

        t0 = time.time()
        pages = ocr_pages(images, corpus_ids)
        self.chunks = chunk_text(pages)
        self.chunk_embeddings = embed_chunks(self.chunks, self.model_name)
        elapsed = time.time() - t0

        print(f"  Indexed in {elapsed:.1f}s")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-k chunks for a query."""
        assert self.chunks is not None, "Call index_pages() first"
        return search_chunks(query, self.chunk_embeddings,
                             self.chunks, self.model, top_k)

    def batch_search(self, queries: list[str], query_ids: list[str],
                     top_k: int = 10) -> dict[str, dict[str, float]]:
        """Batch search returning {query_id: {corpus_id: score}}.

        Maps chunk scores back to page corpus_ids.
        Takes the max chunk score per page.
        """
        run = {}
        for query, qid in zip(queries, query_ids):
            results = self.search(query, top_k=top_k * 3)  # over-retrieve, then dedupe pages
            # Deduplicate: keep max score per corpus_id
            page_scores = {}
            for r in results:
                cid = r["corpus_id"]
                if cid not in page_scores or r["score"] > page_scores[cid]:
                    page_scores[cid] = r["score"]
            # Keep top-k pages
            sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            run[qid] = dict(sorted_pages)
        return run

    def measure_latency(self, queries: list[str],
                        n_queries: int = 100) -> dict:
        """Measure retrieval latency stats."""
        n = min(n_queries, len(queries))
        latencies = []

        for q in queries[:n]:
            t0 = time.perf_counter()
            self.search(q, top_k=5)
            latencies.append((time.perf_counter() - t0) * 1000)

        lat = np.array(latencies)
        return {
            "n_queries": n,
            "n_chunks": len(self.chunks),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)),
            "mean_ms": float(lat.mean()),
        }


# --- Self-check ---

if __name__ == "__main__":
    print("Testing OCR pipeline with synthetic data...")

    # Create a simple test image with text
    from PIL import ImageDraw, ImageFont

    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Revenue Q1 2024: $1.2B\nRevenue Q2 2024: $1.4B", fill="black")
    draw.text((50, 200), "Operating margin improved\nfrom 12% to 15% year-over-year.", fill="black")

    # Test OCR
    try:
        text = ocr_page(img)
        print(f"OCR output ({len(text)} chars): {text[:100]}...")
    except Exception as e:
        print(f"OCR failed (is Tesseract installed?): {e}")
        print("Install: https://github.com/UB-Mannheim/tesseract/wiki")
        exit(1)

    # Test chunking
    pages = [{"corpus_id": "test_page_0", "text": text}]
    chunks = chunk_text(pages)
    assert len(chunks) > 0, "Expected at least one chunk"

    # Test embedding
    embeddings = embed_chunks(chunks)
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] > 0

    # Test search
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    results = search_chunks("What was the revenue?", embeddings, chunks, model, top_k=1)
    assert len(results) == 1
    print(f"\nSearch result: score={results[0]['score']:.3f}")
    print(f"  Text: {results[0]['text'][:100]}...")

    print("\nOK ocr_baseline.py self-check passed")
