"""Streamlit app: Vision RAG vs OCR RAG comparison dashboard."""

import json
import os

import streamlit as st

st.set_page_config(
    page_title="Vision RAG vs OCR RAG",
    page_icon="📄",
    layout="wide",
)

# --- Load eval data ---

@st.cache_data
def load_eval_data():
    """Load all available evaluation results."""
    vision = {}
    ocr = {}

    vision_path = "eval/vidore_vision_results.json"
    if os.path.exists(vision_path):
        with open(vision_path) as f:
            vision = json.load(f)

    for key in ["hr", "computer_science", "finance_en"]:
        ocr_path = f"eval/ocr_baseline_{key}.json"
        if os.path.exists(ocr_path):
            with open(ocr_path) as f:
                ocr[key] = json.load(f)

    return vision, ocr


vision_results, ocr_results = load_eval_data()


# --- Sidebar ---

st.sidebar.title("📄 Vision RAG vs OCR RAG")
st.sidebar.markdown("Head-to-head comparison on ViDoRe v3")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Benchmark Results", "🔍 How It Works", "⚡ Live Demo"],
)

# --- Page: Overview ---

if page == "🏠 Overview":
    st.title("Vision RAG vs OCR RAG")
    st.markdown("### Skipping OCR entirely retrieves the right document page 3–4× more often")

    st.markdown("---")

    # Hero metrics
    col1, col2, col3, col4 = st.columns(4)

    hr_vision = vision_results.get("hr", {})
    hr_ocr = ocr_results.get("hr", {})

    with col1:
        v = hr_vision.get("ndcg_cut_5", 0)
        st.metric("Vision NDCG@5", f"{v:.4f}")
    with col2:
        o = hr_ocr.get("ndcg_cut_5", 0)
        st.metric("OCR NDCG@5", f"{o:.4f}")
    with col3:
        if v and o:
            st.metric("Vision Advantage", f"{v/o:.1f}×")
        else:
            st.metric("Vision Advantage", "—")
    with col4:
        st.metric("DocPruner Compression", "2.0×")

    st.markdown("---")

    # The problem
    st.markdown("### The Problem")
    st.markdown("""
    Every document AI pipeline does: **PDF → OCR → text → search → answer.**

    OCR destroys the most valuable content:
    - **Tables** — cell boundaries lost, columns merge, numbers misread
    - **Charts** — completely invisible to text extraction
    - **Equations** — LaTeX-level structure flattened to gibberish

    Financial reports, research papers, medical documents — the pages that matter most
    are the pages OCR handles worst.
    """)

    # The solution
    st.markdown("### The Solution")
    st.markdown("""
    Skip OCR entirely. Treat every PDF page as an **image**:

    1. **Embed** each page with ColQwen2.5 → ~1024 patch vectors per page
    2. **Prune** to top 50% patches by L2 norm → 2× compression
    3. **Retrieve** with MaxSim late interaction → top-5 pages
    4. **Answer** with Gemini 2.5 Flash → answer + bounding box evidence
    """)

    # Architecture diagram
    if os.path.exists("assets/architecture.svg"):
        st.markdown("### Architecture")
        st.image("assets/architecture.svg", use_container_width=True)


# --- Page: Benchmark Results ---

elif page == "📊 Benchmark Results":
    st.title("📊 Benchmark Results")
    st.markdown("Evaluated on ViDoRe v3 with `pytrec_eval`. Same queries, same answer model — only retrieval differs.")

    if not vision_results and not ocr_results:
        st.warning("No evaluation results found in `eval/`. Run the evaluation first.")
        st.stop()

    # Dataset selector
    available = sorted(set(list(vision_results.keys()) + list(ocr_results.keys())))
    if not available:
        st.info("No datasets evaluated yet.")
        st.stop()

    st.markdown("---")

    # Comparison table
    st.markdown("### Head-to-Head Comparison")

    metrics_display = [
        ("ndcg_cut_5", "NDCG@5"),
        ("ndcg_cut_10", "NDCG@10"),
        ("recall_5", "Recall@5"),
        ("recall_10", "Recall@10"),
        ("map", "MAP"),
        ("retrieval_p50_ms", "Retrieval p50 (ms)"),
        ("retrieval_p95_ms", "Retrieval p95 (ms)"),
    ]

    for dataset_key in available:
        v = vision_results.get(dataset_key, {})
        o = ocr_results.get(dataset_key, {})

        st.markdown(f"#### {dataset_key.replace('_', ' ').title()}")

        cols = st.columns([2, 1, 1, 1])
        cols[0].markdown("**Metric**")
        cols[1].markdown("**Vision RAG**")
        cols[2].markdown("**OCR RAG**")
        cols[3].markdown("**Difference**")

        for metric_key, metric_label in metrics_display:
            v_val = v.get(metric_key)
            o_val = o.get(metric_key)

            cols = st.columns([2, 1, 1, 1])
            cols[0].write(metric_label)

            if v_val is not None:
                cols[1].write(f"{v_val:.4f}" if v_val < 10 else f"{v_val:.1f}")
            else:
                cols[1].write("—")

            if o_val is not None:
                cols[2].write(f"{o_val:.4f}" if o_val < 10 else f"{o_val:.1f}")
            else:
                cols[2].write("—")

            if v_val is not None and o_val is not None and o_val > 0:
                diff = v_val - o_val
                if "latency" in metric_key or "ms" in metric_key:
                    # Lower is better for latency
                    cols[3].write(f"{diff:+.1f} ms")
                else:
                    ratio = v_val / o_val
                    cols[3].write(f"{ratio:.1f}×")
            else:
                cols[3].write("—")

        st.markdown("---")

    # Bar chart visualization
    st.markdown("### NDCG@5 Comparison")

    chart_data = []
    for dataset_key in available:
        v = vision_results.get(dataset_key, {})
        o = ocr_results.get(dataset_key, {})
        v_ndcg = v.get("ndcg_cut_5", 0)
        o_ndcg = o.get("ndcg_cut_5", 0)
        if v_ndcg or o_ndcg:
            chart_data.append({
                "Dataset": dataset_key.replace("_", " ").title(),
                "Vision RAG": v_ndcg,
                "OCR RAG": o_ndcg,
            })

    if chart_data:
        import pandas as pd
        df = pd.DataFrame(chart_data).set_index("Dataset")
        st.bar_chart(df, color=["#4A90D9", "#D94A4A"])

    # Dataset details
    st.markdown("### Dataset Details")
    for dataset_key in available:
        v = vision_results.get(dataset_key, {})
        o = ocr_results.get(dataset_key, {})
        data = v or o

        with st.expander(f"{dataset_key.replace('_', ' ').title()}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Pages", data.get("n_pages", "—"))
            c2.metric("Queries", data.get("n_queries", "—"))
            if v:
                c3.metric("Compression", f"{v.get('compression_ratio', 0):.1f}×")


# --- Page: How It Works ---

elif page == "🔍 How It Works":
    st.title("🔍 How It Works")

    st.markdown("---")

    # Vision RAG pipeline
    st.markdown("### Pipeline A: Vision RAG")
    st.markdown("""
    ```
    PDF page (image, 180 DPI)
         │
         ▼
    ColQwen2.5 encoder
    → ~1024 patch vectors per page (128-dim each)
    → each patch = a region of the page
         │
         ▼
    DocPruner (L2-norm top 50%)
    → drop lowest-norm patches (background/whitespace)
    → 2× compression, minimal accuracy loss
         │
         ▼
    MaxSim retrieval (pure numpy)
    → for each query token: max similarity across all patches
    → sum across query tokens = page score
    → top-5 pages returned
         │
         ▼
    Gemini 2.5 Flash
    → receives page images directly
    → returns: answer + box_2d evidence regions (bounding boxes)
    ```
    """)

    st.markdown("#### MaxSim in 5 Lines")
    st.code("""# For each page:
sims = query_embedding @ page_embedding.T   # (q_tokens, n_patches)
score = sims.max(axis=1).sum()              # max per query token, then sum

# That's it. Late interaction = token-level matching.""", language="python")

    st.markdown("---")

    # OCR RAG pipeline
    st.markdown("### Pipeline B: OCR RAG")
    st.markdown("""
    ```
    PDF page (image, 180 DPI)
         │
         ▼
    PyTesseract OCR
    → raw text extraction
         │
         ▼
    Layout-aware chunking
    → split on paragraph boundaries
    → 1500 char max, 200 char overlap
         │
         ▼
    sentence-transformers (all-MiniLM-L6-v2)
    → single 384-dim vector per chunk
         │
         ▼
    Cosine similarity retrieval
    → top-5 chunks returned
         │
         ▼
    Gemini 2.5 Flash
    → receives text excerpts
    → returns: answer + cited chunk indices
    ```
    """)

    st.markdown("---")

    # Key differences
    st.markdown("### Why Vision Wins on Structured Content")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Vision RAG ✅")
        st.markdown("""
        - Sees the **actual table** — rows, columns, cell alignment
        - Sees **charts** — bars, lines, legends, axes
        - Sees **equations** — symbols, fractions, subscripts
        - Preserves **spatial layout** through patch positions
        - ~1024 vectors capture page structure
        """)
    with col2:
        st.markdown("#### OCR RAG ❌")
        st.markdown("""
        - Table becomes **flat text** — "Revenue 1.2B 1.4B 1.6B"
        - Charts produce **no text at all**
        - Equations become **"E = mc 2"** or worse
        - Layout **destroyed** — columns merge, headers detach
        - Single vector compresses everything into 1 point
        """)

    st.markdown("---")

    st.markdown("### DocPruner: Compression Without Pain")
    st.markdown("""
    ColQwen2.5 produces ~1024 patches per page. Many are background, whitespace, or
    low-information margins. DocPruner ranks patches by L2 norm and keeps the top 50%.

    **Result:** 2× storage compression. The pruned index retrieves almost as accurately
    as the full index because the discarded patches carried little semantic information.
    """)

    # Pruning stats
    for key in ["hr", "computer_science"]:
        v = vision_results.get(key, {})
        if v:
            raw = v.get("raw_patches", 0)
            pruned = v.get("pruned_patches", 0)
            if raw and pruned:
                st.markdown(f"**{key.replace('_', ' ').title()}:** "
                           f"{raw:,} → {pruned:,} patches "
                           f"({pruned/raw:.0%} kept, "
                           f"{v.get('compression_ratio', 0):.1f}× compression)")


# --- Page: Live Demo ---

elif page == "⚡ Live Demo":
    st.title("⚡ Live Demo")

    # Check for embeddings
    has_embeddings = os.path.exists("embeddings")
    has_api_key = bool(os.environ.get("GOOGLE_API_KEY"))

    if not has_embeddings:
        st.warning(
            "**Embeddings not found.** Generate them first using `notebooks/kaggle_embed.ipynb` "
            "or `notebooks/colab_embed.ipynb`, then place them in `embeddings/`."
        )
        st.markdown("""
        ```
        embeddings/
        ├── hr/
        │   ├── pages_pruned.npz
        │   ├── pages_pruned_meta.json
        │   ├── queries.npz
        │   └── queries_meta.json
        └── ...
        ```
        """)

    if not has_api_key:
        st.info(
            "**Set `GOOGLE_API_KEY`** environment variable for Gemini answering. "
            "Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)"
        )

    if has_embeddings:
        # Dataset selector
        embed_datasets = [d for d in os.listdir("embeddings")
                         if os.path.isdir(os.path.join("embeddings", d))]

        if not embed_datasets:
            st.error("Embeddings directory is empty.")
            st.stop()

        dataset = st.selectbox("Dataset", embed_datasets)

        # Load index
        from src.index import build_index, load_embeddings

        pruned_path = f"embeddings/{dataset}/pages_pruned.npz"
        queries_path = f"embeddings/{dataset}/queries.npz"

        if not os.path.exists(pruned_path):
            st.error(f"Missing: {pruned_path}")
            st.stop()

        with st.spinner("Loading index..."):
            index = build_index("embeddings", dataset, pruned=True)

        st.success(f"Index loaded: {index.n_pages} pages")

        # Query input
        query = st.text_input("Ask a question about the documents:",
                              placeholder="e.g., What was the total revenue in Q3?")

        if query:
            # Embed the query
            st.markdown("#### Retrieval Results")

            if os.path.exists(queries_path):
                query_embs, query_meta = load_embeddings(queries_path)

                # Check if this is a pre-embedded query
                match = None
                for i, qm in enumerate(query_meta):
                    if qm.get("text", "").strip().lower() == query.strip().lower():
                        match = i
                        break

                if match is not None:
                    results = index.search(query_embs[match], top_k=5)
                    for r in results:
                        st.markdown(f"- **{r['corpus_id']}** — score: {r['score']:.2f}")

                    # Gemini answering
                    if has_api_key:
                        st.markdown("#### Answer")
                        st.info("Gemini answering requires page images. "
                                "Load the dataset with `src/ingest.py` to enable.")
                else:
                    st.warning(
                        "Custom queries require the ColQwen2.5 model to embed. "
                        "Try one of the benchmark queries from the dataset, or run "
                        "`src/embed.py` locally with a GPU."
                    )

                # Show some example queries
                with st.expander("📋 Example queries from the benchmark"):
                    for qm in query_meta[:20]:
                        text = qm.get("text", "")
                        if text:
                            st.markdown(f"- {text}")
            else:
                st.warning(f"Query embeddings not found at {queries_path}")

    else:
        # Show what the demo would look like
        st.markdown("---")
        st.markdown("### Demo Preview")
        st.markdown("""
        When embeddings are loaded, this page provides:

        1. **Query input** — type a question or pick from benchmark queries
        2. **Retrieval results** — top-5 pages ranked by MaxSim score
        3. **Gemini answer** — structured answer with confidence score
        4. **Evidence overlay** — bounding boxes drawn on source page images

        Generate embeddings first (see instructions above), then restart the app.
        """)
