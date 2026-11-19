# Multimodal Document QA — Build Plan

> Head-to-head: Vision RAG (ColQwen2.5) vs OCR RAG on ViDoRe v3 benchmarks.
> Prove that skipping OCR beats it on tables, charts, and equations — with numbers.

---

## The Problem

Every enterprise document pipeline: PDF → OCR → text → search → answer.

OCR **destroys** the most valuable content:
- **Tables** — cell boundaries lost, columns merge, numbers misread
- **Charts** — completely invisible to text extraction
- **Equations** — LaTeX-level structure flattened to gibberish
- **Complex layouts** — multi-column, sidebars, headers bleed together

This isn't a minor issue. Financial 10-Ks, research papers, medical reports — the pages that matter most are the pages OCR handles worst.

## The Solution

Skip OCR entirely. Treat every PDF page as an **image**:

1. **Embed** each page with ColQwen2.5 → ~1024 patch vectors per page (multi-vector)
2. **Prune** to top 50% patches by L2 norm → 2x compression
3. **Retrieve** with MaxSim late interaction → top-5 pages
4. **Answer** with Gemini 2.5 Flash → answer + bounding box evidence regions
5. **Compare** against a fair OCR baseline on the same benchmarks
6. **Show** results in a Streamlit viewer with evidence overlays

---

## Rubric (what gets graded)

| Weight | Criterion | How We Hit It |
|--------|-----------|---------------|
| 25% | ViDoRe v3 / M3DocVQA accuracy | `vidore-benchmark` on 3 English datasets, NDCG@5 vision vs OCR vs published leaderboard |
| 20% | Evidence-region grounding | Gemini box_2d predictions, IoU > 0.3 against ViDoRe qrel bounding boxes |
| 20% | Storage & latency engineering | DocPruner compression ratio, p95 retrieval + answer latency |
| 20% | Multi-page reasoning | 100 curated questions requiring 2+ pages, EM + token-F1 |
| 15% | Source-inspection UX | Streamlit: side-by-side comparison, evidence overlays, benchmark dashboard |

---

## Architecture

```
                    ┌──────────────────────────────────┐
                    │         PDF Corpus / ViDoRe v3    │
                    └──────────┬───────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
     Pipeline A: Vision RAG          Pipeline B: OCR RAG
     ──────────────────────          ─────────────────────
     page images (180 DPI)           page images → PyTesseract
           │                                │
     ColQwen2.5 embed                  text chunks (layout-aware)
     (multi-vector, ~1024 patches)         │
           │                         sentence-transformers
     DocPruner (top 50% L2 norm)     (single-vector, 384 dim)
           │                                │
     MaxSim retrieval (numpy)        cosine similarity (numpy)
           │                                │
     top-5 page images               top-5 text chunks
           │                                │
           └──────────┬────────────────────┘
                      ▼
              Gemini 2.5 Flash
              (answer + box_2d evidence)
                      │
                      ▼
              Evaluation Framework
              (NDCG@5, grounding F1, latency, multi-page)
                      │
                      ▼
              Streamlit Viewer
              (side-by-side + evidence overlays + dashboard)
```

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Vector DB | **Pure numpy** (no Qdrant) | 5,400 pages × 512 patches = ~1.4 GB. Fits in 31 GB RAM. MaxSim is 5 lines of einsum. Recruiter sees the math. |
| GPU | **Kaggle P100** (free tier) | No local NVIDIA GPU. P100 embeds 500 pages in ~30 min. 30 GPU hours/week. |
| ViDoRe datasets | **3 English** (finance, CS, HR) | ~5,400 pages total. ~4.5h on P100. Enough diversity, within GPU budget. |
| OCR baseline | **PyTesseract + layout-aware chunking** | Fair baseline — not a straw man, not overbuilt. |
| Same LLM for both | **Gemini 2.5 Flash** for both pipelines | Isolates the only variable: retrieval method + context format. |
| Requirements split | **Two files** (local + Kaggle) | Avoids 2 GB PyTorch install on a machine with no GPU. |

---

## Build Steps

### Step 1: Setup + Ingest (~2h)
- [ ] Create venv, install local requirements
- [ ] Install Tesseract binary on Windows
- [ ] Write `src/ingest.py` — ViDoRe dataset loader + PyMuPDF renderer
- [ ] Test: load `vidore_v3_finance_en`, verify images + queries + qrels

### Step 2: OCR Baseline (~2h)
- [ ] Write `src/ocr_baseline.py` — PyTesseract → chunking → sentence-transformers → cosine retrieval
- [ ] Test: OCR 50 pages, embed chunks, query and get results

### Step 3: Embedding on Kaggle (~4h active + GPU wait)
- [ ] Write `src/embed.py` and `src/prune.py` locally
- [ ] Build `notebooks/kaggle_embed.ipynb`
- [ ] Run on Kaggle: embed 3 ViDoRe datasets with ColQwen2.5
- [ ] Run DocPruner, save raw + pruned .npz
- [ ] Download embeddings to local `embeddings/`

### Step 4: MaxSim Retrieval (~2h)
- [ ] Write `src/index.py` — MaxSimIndex with numpy einsum
- [ ] Load pruned embeddings, build index, test queries
- [ ] Measure p95 retrieval latency

### Step 5: Gemini Answering + Evidence (~2h)
- [ ] Write `src/answer.py` — vision path (images + box_2d) and OCR path (text)
- [ ] Write overlay renderer for bounding boxes
- [ ] Test end-to-end both pipelines

### Step 6: Multi-page Questions (~2h)
- [ ] Write `src/generate_multipage.py`
- [ ] Generate ~150 candidates with Gemini
- [ ] **You curate** to 100 in `eval/questions.json`

### Step 7: Full Evaluation (~3h)
- [ ] Write `src/evaluate.py` — all rubric metrics
- [ ] Run vidore-benchmark on Kaggle for official NDCG@5
- [ ] Run grounding, latency, multi-page eval locally
- [ ] Write `eval/results.json`

### Step 8: Streamlit Demo (~3h)
- [ ] Write `app.py` — 3 tabs: Ask Question, Benchmark Results, Evidence Inspector
- [ ] Side-by-side vision vs OCR comparison
- [ ] Evidence overlay rendering

### Step 9: Polish + Ship (~2-3h)
- [ ] Update README with results + architecture diagram + demo GIF
- [ ] Record demo GIF
- [ ] Push to public GitHub
- [ ] Post on LinkedIn + X

---

## File Structure

```
multimodal-doc-qa-capstone/
├── PLAN.md                      ← you are here
├── README.md                    ← recruiter-facing overview
├── requirements.txt             ← local deps (no PyTorch)
├── requirements-kaggle.txt      ← Kaggle GPU deps
├── notebooks/
│   └── kaggle_embed.ipynb       ← all GPU work
├── src/
│   ├── ingest.py                ← ViDoRe loader + PyMuPDF renderer
│   ├── embed.py                 ← ColQwen2.5 embedding
│   ├── prune.py                 ← DocPruner L2-norm top-k
│   ├── index.py                 ← MaxSimIndex (pure numpy)
│   ├── answer.py                ← Gemini + box_2d evidence + overlays
│   ├── ocr_baseline.py          ← OCR pipeline end-to-end
│   ├── evaluate.py              ← all rubric metrics
│   └── generate_multipage.py    ← multi-page question generator
├── app.py                       ← Streamlit viewer (3 tabs)
├── eval/
│   ├── questions.json           ← 100 curated multi-page questions
│   └── results.json             ← full comparison matrix
├── embeddings/                  ← .npz from Kaggle
├── corpus/                      ← optional own PDFs
│   └── rendered/
└── assets/
    └── architecture.png
```

---

## Tech Stack

```
Python 3.12
PyMuPDF (fitz)              — PDF → page images
colpali-engine              — ColQwen2.5 embeddings (Kaggle only)
google-genai                — Gemini 2.5 Flash API
streamlit                   — demo UI
pytesseract + tesseract     — OCR baseline
sentence-transformers       — text embeddings for OCR pipeline
numpy                       — MaxSim, array ops, the entire index
Pillow                      — image handling
datasets                    — HuggingFace ViDoRe loader
vidore-benchmark            — official ViDoRe evaluation (Kaggle only)
```
