# Multimodal Document QA: Vision RAG vs OCR RAG

A head-to-head comparison of **vision-first retrieval** (ColQwen2.5) vs **traditional OCR retrieval** (PyTesseract + sentence-transformers) on the ViDoRe v3 benchmark, with evidence-grounded answers from Gemini 2.5 Flash.

---

## Why This Exists

Every enterprise document pipeline does: **PDF → OCR → text → search → answer.**

OCR destroys the most valuable content:

- **Tables** — cell boundaries lost, columns merge, numbers misread
- **Charts** — completely invisible to text extraction
- **Equations** — LaTeX-level structure flattened to gibberish
- **Complex layouts** — multi-column, sidebars, headers bleed together

Financial 10-Ks, research papers, medical reports — the pages that matter most are the pages OCR handles worst. This project proves the difference with real numbers on a real benchmark.

---

## What This Project Does

Two complete retrieval pipelines evaluated on the same benchmark, same questions, same answer model — so the only variable is the retrieval method.

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
              (answer + box_2d evidence regions)
                      │
                      ▼
              Evaluation (NDCG@5, Recall, MAP, latency)
```

**Pipeline A (Vision RAG):** Treat every PDF page as an image. Embed with ColQwen2.5 (multi-vector, ~1024 patch tokens per page). Prune to top 50% patches by L2 norm. Retrieve with MaxSim late interaction. Answer with Gemini 2.5 Flash using the page images directly — it returns structured JSON with bounding-box evidence regions.

**Pipeline B (OCR RAG):** Run PyTesseract on each page image. Chunk the text with layout-aware paragraph splitting. Embed chunks with `all-MiniLM-L6-v2`. Retrieve with cosine similarity. Same Gemini model answers, but from text excerpts instead of images.

---

## Results

Evaluated on ViDoRe v3 (HR and Computer Science splits, ~2,470 pages, ~3,198 queries):

| Metric | Vision RAG | OCR RAG | Vision wins by |
|--------|-----------|---------|----------------|
| **NDCG@5 (HR)** | **0.4232** | 0.1261 | **3.4×** |
| Recall@5 (HR) | **0.4092** | 0.1198 | 3.4× |
| Recall@10 (HR) | **0.5365** | 0.1710 | 3.1× |
| MAP (HR) | **0.3302** | 0.0894 | 3.7× |
| NDCG@5 (CS) | **0.0700** | — | — |
| Retrieval p95 (HR) | 139ms | 32ms | OCR faster |
| DocPruner compression | 2.0× | N/A | — |

Vision RAG retrieves the right page **3–4× more often** than OCR on documents with tables, charts, and complex layouts. OCR is faster per query (single-vector cosine vs multi-vector MaxSim), but retrieves the wrong content.

---

## How To Run

### Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and in PATH
- A [Google AI API key](https://aistudio.google.com/apikey) (for Gemini answering)

### 1. Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate Embeddings

Embeddings are not included in this repo (~500 MB per dataset). They must be generated using a GPU.

**Option A — Kaggle (free, recommended):**
1. Upload `notebooks/kaggle_embed.ipynb` to [Kaggle](https://www.kaggle.com/)
2. Enable GPU P100 in Settings → Accelerator
3. Set `DATASET_INDEX` (0=HR, 1=CS, 2=Finance) and Run All
4. Download the output zip
5. Unzip into `embeddings/{dataset_key}/`

**Option B — Google Colab (free):**
1. Upload `notebooks/colab_embed.ipynb` to Colab
2. Change runtime to T4 GPU
3. Set `DATASET_INDEX` and Run All
4. Download the zip, unzip into `embeddings/{dataset_key}/`

**Option C — Local GPU (if available):**
```bash
pip install -r requirements-colab.txt   # adds colpali-engine + torch
python src/embed.py hr                  # or finance_en, computer_science, all
```

After embedding, the `embeddings/` folder should look like:
```
embeddings/
├── hr/
│   ├── pages.npz              # raw page embeddings
│   ├── pages_pruned.npz       # pruned (50%) page embeddings
│   ├── queries.npz            # query embeddings
│   ├── qrels.json             # relevance judgments
│   └── results.json           # evaluation metrics
├── computer_science/
│   └── ...
└── finance_en/
    └── ...
```

### 3. Run OCR Baseline Evaluation

```bash
python src/run_ocr_eval.py hr
```

### 4. Try Queries (requires embeddings + API key)

```bash
export GOOGLE_API_KEY=your-key-here
# or on Windows: set GOOGLE_API_KEY=your-key-here
```

---

## Why Embeddings Are Required

ColQwen2.5 encodes each PDF page into ~1024 patch vectors (128-dim each). These multi-vector embeddings are what make MaxSim retrieval work — each patch represents a region of the page (a cell in a table, a bar in a chart, a line of an equation), and retrieval matches query tokens to the most relevant patches.

**Not included in this repo:** The three ViDoRe v3 datasets produce ~500 MB of embeddings (compressed). Too large for Git, and regenerating them is straightforward with the provided notebooks.

**Free cloud GPUs are sufficient:** ColQwen2.5 needs ~5 GB VRAM. Embedding ~5,400 pages takes ~1.5h on Kaggle P100 or ~4h on Colab T4. No local GPU required.

**How a production system would differ:**

| This project | Production |
|-------------|-----------|
| Pre-computed `.npz` files on disk | Vector database (Qdrant, Weaviate, Pinecone) |
| Embed once, evaluate offline | Embed on ingest, serve retrieval via API |
| Full numpy MaxSim scan | ANN index (HNSW) for sub-10ms retrieval |
| Free-tier GPU notebooks | Dedicated GPU inference server or embedding API |
| ~5K pages | Millions of pages with sharding + replication |

The core approach (vision embedding → multi-vector retrieval → VLM answering) stays the same. The infrastructure changes, not the architecture.

---

## Step-by-Step Guide: Reproduce This Project

### Step 1: Understand the Datasets

This project uses [ViDoRe v3](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d) — a benchmark for visual document retrieval. It provides:
- **Corpus**: PDF pages as images (already rendered)
- **Queries**: Natural-language questions about those pages
- **Qrels**: Ground-truth relevance judgments (which page answers which query)

Three English splits are used:

| Dataset | Pages | Queries | Content type |
|---------|-------|---------|-------------|
| `vidore/vidore_v3_hr` | 1,110 | 1,908 | HR documents |
| `vidore/vidore_v3_computer_science` | 1,360 | 1,290 | CS papers |
| `vidore/vidore_v3_finance_en` | 2,942 | 3,000+ | Financial reports |

### Step 2: Generate Vision Embeddings (Kaggle)

ColQwen2.5 is the vision encoder. It takes a page image and produces ~1024 **patch vectors** (128-dim each). Each patch corresponds to a spatial region of the page — a table cell, a chart bar, a paragraph. This multi-vector representation is what enables token-level matching during retrieval.

**Why Kaggle?** ColQwen2.5 needs ~5 GB VRAM. Kaggle provides a free P100 GPU (16 GB VRAM, 30 hours/week). No credit card needed.

**How to run on Kaggle:**

1. Go to [kaggle.com](https://www.kaggle.com/) and create an account
2. Click **New Notebook** (top right)
3. Upload `notebooks/kaggle_embed.ipynb` (File → Import Notebook)
4. In the right sidebar: **Settings → Accelerator → GPU P100**
5. Set `DATASET_INDEX = 0` in the first code cell (0=HR, the smallest dataset — start here)
6. Click **Run All**
7. Wait ~17 minutes for HR, ~21 min for CS, ~45 min for Finance
8. When done, go to **Output** tab (top of page) → download the zip
9. Unzip into `embeddings/hr/` locally
10. Change `DATASET_INDEX` to 1 → **Runtime → Restart & Run All** → repeat for each dataset

**What gets generated:**
- `pages.npz` — raw embeddings (~1024 patches per page)
- `pages_pruned.npz` — pruned to top 50% patches by L2 norm (DocPruner)
- `queries.npz` — query token embeddings
- `qrels.json` — relevance judgments from the dataset
- `results.json` — NDCG@5, Recall, MAP, latency metrics

The notebook also runs the full evaluation (MaxSim retrieval + pytrec_eval metrics) on Kaggle itself, so the results are ready to compare.

### Step 3: Set Up the OCR Baseline Locally

The OCR pipeline runs entirely on CPU. It needs Tesseract (the OCR engine) and sentence-transformers (for text embeddings).

**Install Tesseract:**

- **Windows**: Download from [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki). Run the installer. Add to PATH or note the install path.
- **Mac**: `brew install tesseract`
- **Linux**: `sudo apt install tesseract-ocr`

**Why Tesseract?** It is the most widely-used open-source OCR engine. Choosing it over commercial alternatives keeps the OCR baseline fair and reproducible. The point is not to use the best possible OCR — the point is to show that even reasonable OCR loses structural information that vision retrieval preserves.

**Why sentence-transformers (`all-MiniLM-L6-v2`)?** A standard lightweight text embedding model (384-dim, single-vector). Fast, well-benchmarked, runs on CPU. The OCR pipeline uses single-vector cosine similarity — the traditional approach — to contrast with ColQwen2.5's multi-vector MaxSim.

**Install and run:**

```bash
python -m venv venv && venv\Scripts\activate  # or source venv/bin/activate
pip install -r requirements.txt

# Run OCR evaluation on the HR dataset
python src/run_ocr_eval.py hr
```

This takes ~40 minutes on the HR split (OCR is slow). It will:
1. Download the ViDoRe dataset from HuggingFace
2. OCR every page with Tesseract
3. Chunk the text (layout-aware, 1500 char max)
4. Embed chunks with sentence-transformers
5. Run cosine similarity retrieval for all queries
6. Compute NDCG@5, Recall, MAP with pytrec_eval
7. Save results to `eval/ocr_baseline_hr.json`

### Step 4: Compare Results

After both pipelines have run, the results are in `eval/`:

```bash
# Vision results (from Kaggle)
cat eval/vidore_vision_results.json

# OCR results (from local run)
cat eval/ocr_baseline_hr.json
```

The `run_ocr_eval.py` script also prints a side-by-side comparison table automatically if vision results exist.

### Step 5: Run the Streamlit Dashboard

```bash
pip install streamlit  # if not already installed
streamlit run app.py
```

The dashboard shows the benchmark results, architecture explanation, and (when embeddings are present) a live query interface.

### References

- **ColPali / ColQwen2.5**: [Faysse et al., 2024 — ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01272)
- **ViDoRe Benchmark**: [vidore-benchmark on HuggingFace](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d)
- **MaxSim / Late Interaction**: [Khattab & Zaharia, 2020 — ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction](https://arxiv.org/abs/2004.12832)
- **DocPruner concept**: Inspired by token pruning in vision transformers — patches with low L2 norm carry less semantic information
- **pytrec_eval**: [Python wrapper for trec_eval](https://github.com/cvangysel/pytrec_eval) — standard IR evaluation metrics
- **Tesseract OCR**: [tesseract-ocr on GitHub](https://github.com/tesseract-ocr/tesseract)
- **sentence-transformers**: [SBERT.net](https://www.sbert.net/) — pre-trained text embedding models

---

## Project Structure

```
multimodal-doc-qa-capstone/
├── src/
│   ├── embed.py           # ColQwen2.5 embedding (CPU or GPU)
│   ├── prune.py           # DocPruner: L2-norm top-k patch selection
│   ├── index.py           # MaxSimIndex: pure-numpy late-interaction retrieval
│   ├── answer.py          # Gemini 2.5 Flash answering + box_2d evidence overlay
│   ├── ingest.py          # ViDoRe v3 dataset loader + PDF renderer (PyMuPDF)
│   ├── ocr_baseline.py    # Full OCR pipeline: Tesseract → chunk → embed → retrieve
│   └── run_ocr_eval.py    # OCR evaluation harness with pytrec_eval
├── notebooks/
│   ├── kaggle_embed.ipynb # GPU embedding on Kaggle P100 (~1.5h for all datasets)
│   └── colab_embed.ipynb  # GPU embedding on Colab T4 (~4h for all datasets)
├── eval/
│   ├── vidore_vision_results.json  # Vision RAG metrics (NDCG, Recall, MAP, latency)
│   └── ocr_baseline_hr.json       # OCR baseline metrics
├── embeddings/            # NOT IN REPO — generated via notebooks (see above)
├── corpus/                # Optional: your own PDFs for local testing
│   └── rendered/          # Page images rendered by ingest.py
├── requirements.txt       # Local deps (no PyTorch)
└── requirements-colab.txt # GPU deps (colpali-engine + torch)
```

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Visual embedding | ColQwen2.5 (multi-vector) | SOTA late-interaction model for document retrieval |
| Pruning | L2-norm top 50% | 2× compression, minimal accuracy loss |
| Retrieval | MaxSim (pure numpy) | 5 lines of einsum, no vector DB needed at this scale |
| Answering | Gemini 2.5 Flash | Vision-capable, structured JSON output with box_2d |
| OCR baseline | PyTesseract + all-MiniLM-L6-v2 | Fair comparison — layout-aware chunking, not a straw man |
| PDF rendering | PyMuPDF | Fast, 180 DPI page images |
| Evaluation | pytrec_eval | Standard IR metrics (NDCG, Recall, MAP) |
| GPU compute | Kaggle P100 / Colab T4 (free) | No local GPU required |
