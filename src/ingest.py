"""Ingest: load ViDoRe v3 datasets + render local PDFs to page images."""

import json
import os
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


# --- ViDoRe v3 dataset loading ---

def load_vidore_dataset(dataset_name: str) -> dict:
    """Load a ViDoRe v3 dataset from HuggingFace.
    Returns dict with 'corpus', 'queries', 'qrels' splits as lists of dicts.

    Lazy import of datasets to avoid pulling it in when not needed.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, trust_remote_code=True)

    result = {}
    for split_name in ds:
        rows = []
        for row in ds[split_name]:
            rows.append(dict(row))
        result[split_name] = rows

    print(f"Loaded {dataset_name}:")
    for split_name, rows in result.items():
        print(f"  {split_name}: {len(rows)} rows")

    return result


def get_corpus_images(corpus: list[dict], image_col: str = "image") -> list[Image.Image]:
    """Extract PIL images from a corpus split."""
    return [row[image_col] for row in corpus]


def get_corpus_ids(corpus: list[dict], id_col: str = "corpus_id") -> list[str]:
    """Extract corpus IDs."""
    return [str(row[id_col]) for row in corpus]


def get_queries(queries: list[dict], id_col: str = "query_id",
                text_col: str = "query") -> tuple[list[str], list[str]]:
    """Extract query IDs and texts."""
    ids = [str(row[id_col]) for row in queries]
    texts = [row[text_col] for row in queries]
    return ids, texts


def build_qrels_dict(qrels: list[dict], query_col: str = "query_id",
                     corpus_col: str = "corpus_id",
                     score_col: str = "score") -> dict[str, dict[str, int]]:
    """Build {query_id: {corpus_id: score}} from qrels list."""
    result = {}
    for row in qrels:
        qid = str(row[query_col])
        cid = str(row[corpus_col])
        score = int(row[score_col])
        result.setdefault(qid, {})[cid] = score
    return result


# --- Local PDF rendering ---

def render_pdf(pdf_path: str, dpi: int = 180) -> list[dict]:
    """Render a PDF to page images.
    Returns list of {doc_id, page_num, image: PIL.Image}.
    """
    doc = fitz.open(pdf_path)
    doc_id = Path(pdf_path).stem
    pages = []

    zoom = dpi / 72  # PyMuPDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append({
            "doc_id": doc_id,
            "page_num": page_num,
            "image": img,
        })

    doc.close()
    print(f"Rendered {pdf_path}: {len(pages)} pages at {dpi} DPI")
    return pages


def render_corpus_dir(corpus_dir: str, output_dir: str = None,
                      dpi: int = 180) -> list[dict]:
    """Render all PDFs in a directory. Optionally save images to output_dir."""
    all_pages = []
    pdf_files = sorted(Path(corpus_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {corpus_dir}")
        return all_pages

    for pdf_path in pdf_files:
        pages = render_pdf(str(pdf_path), dpi=dpi)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for page in pages:
                fname = f"{page['doc_id']}_page{page['page_num']:04d}.png"
                page["image"].save(os.path.join(output_dir, fname))
                page["image_path"] = os.path.join(output_dir, fname)

        all_pages.extend(pages)

    print(f"Total: {len(all_pages)} pages from {len(pdf_files)} PDFs")
    return all_pages


# --- Self-check ---

if __name__ == "__main__":
    # Test ViDoRe loading with a small sample
    print("Testing ViDoRe dataset loading...")
    data = load_vidore_dataset("vidore/vidore_v3_finance_en")

    for split_name, rows in data.items():
        print(f"\n{split_name}:")
        if rows:
            print(f"  Columns: {list(rows[0].keys())}")
            row = rows[0]
            for k, v in row.items():
                if isinstance(v, Image.Image):
                    print(f"  {k}: PIL Image {v.size}")
                elif isinstance(v, str) and len(v) > 80:
                    print(f"  {k}: '{v[:80]}...'")
                else:
                    print(f"  {k}: {repr(v)}")

    # Test local PDF rendering if any PDFs exist
    corpus_dir = os.path.join(os.path.dirname(__file__), "..", "corpus")
    if list(Path(corpus_dir).glob("*.pdf")):
        pages = render_corpus_dir(corpus_dir, os.path.join(corpus_dir, "rendered"))
        assert len(pages) > 0, "Expected at least one rendered page"
        assert "image" in pages[0], "Expected 'image' key in page dict"
        print("\nOK Local PDF rendering works")
    else:
        print(f"\nNo PDFs in {corpus_dir} — skipping local render test")

    print("\nOK ingest.py self-check passed")
