"""ColQwen2.5 embedding — runs on CPU or GPU.

Usage:
    python src/embed.py                    # embed HR dataset (smallest, ~6h CPU)
    python src/embed.py finance_en         # embed finance dataset (~16h CPU)
    python src/embed.py computer_science   # embed CS dataset (~7.5h CPU)
    python src/embed.py all                # embed all 3 datasets sequentially

Saves to embeddings/{dataset_key}/ with checkpointing every 100 pages.
Safe to interrupt and restart — picks up where it left off.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image


DATASETS = {
    "hr": "vidore/vidore_v3_hr",
    "computer_science": "vidore/vidore_v3_computer_science",
    "finance_en": "vidore/vidore_v3_finance_en",
}

# ponytail: hardcoded config, no config file needed
MODEL_NAME = "vidore/colqwen2.5-v0.2"
BATCH_SIZE = 1        # CPU-safe batch size
CHECKPOINT_EVERY = 100  # save every N pages
KEEP_FRACTION = 0.5    # DocPruner keep ratio


def load_model():
    """Load ColQwen2.5. Uses GPU if available, CPU otherwise."""
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {MODEL_NAME} on {device}...")
    t0 = time.time()

    model = ColQwen2_5.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    ).to(device).eval()

    processor = ColQwen2_5_Processor.from_pretrained(MODEL_NAME)

    print(f"Model loaded in {time.time()-t0:.0f}s")
    if device == "cuda":
        print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    else:
        print("Running on CPU. This will be slow but it works.")

    return model, processor, device


def embed_pages(model, processor, images, device, batch_size=1):
    """Embed page images. Returns list of (n_patches, dim) numpy arrays."""
    all_embeddings = []
    total = len(images)
    t_start = time.time()

    for i in range(0, total, batch_size):
        batch_imgs = images[i:i + batch_size]
        batch = processor.process_images(batch_imgs)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            embeddings = model(**batch)

        for emb in embeddings:
            all_embeddings.append(emb.cpu().float().numpy())

        del batch, embeddings
        if device == "cuda":
            torch.cuda.empty_cache()

        done = min(i + batch_size, total)
        if done % 10 == 0 or done == total:
            elapsed = time.time() - t_start
            rate = elapsed / done
            eta = rate * (total - done)
            print(f"  {done}/{total} pages  "
                  f"({rate:.1f}s/page, ~{eta/60:.0f}min remaining)")

    return all_embeddings


def embed_queries(model, processor, queries, device, batch_size=4):
    """Embed text queries. Returns list of (n_tokens, dim) numpy arrays."""
    all_embeddings = []

    for i in range(0, len(queries), batch_size):
        batch = processor.process_queries(queries[i:i + batch_size])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            embeddings = model(**batch)
        for emb in embeddings:
            all_embeddings.append(emb.cpu().float().numpy())
        del batch, embeddings

    print(f"  {len(all_embeddings)} queries embedded")
    return all_embeddings


def save_embeddings(embeddings_list, metadata, path):
    """Save as compressed .npz + JSON sidecar."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {f"page_{i}": emb for i, emb in enumerate(embeddings_list)}
    np.savez_compressed(path, **arrays)
    with open(path.replace(".npz", "_meta.json"), "w") as f:
        json.dump(metadata, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved {len(embeddings_list)} embeddings -> {path} ({size_mb:.1f} MB)")


def prune_l2_topk(embeddings_list, keep_fraction=0.5):
    """DocPruner: keep top patches by L2 norm."""
    pruned = []
    for emb in embeddings_list:
        norms = np.linalg.norm(emb, axis=1)
        k = max(1, int(len(norms) * keep_fraction))
        top_idx = np.argpartition(norms, -k)[-k:]
        top_idx.sort()
        pruned.append(emb[top_idx])
    return pruned


def process_dataset(dataset_key, model, processor, device):
    """Full pipeline for one dataset: load -> embed -> prune -> save."""
    from datasets import load_dataset

    dataset_name = DATASETS[dataset_key]
    out_dir = os.path.join("embeddings", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    # --- Check if already done ---
    done_marker = os.path.join(out_dir, "results.json")
    if os.path.exists(done_marker):
        print(f"\n{dataset_key}: already complete (found {done_marker}), skipping")
        return

    print(f"\n{'=' * 60}")
    print(f"Processing: {dataset_key}")
    print(f"{'=' * 60}")

    # --- Load dataset ---
    print(f"\nLoading {dataset_name}...")
    ds = load_dataset(dataset_name, trust_remote_code=True)
    print(f"Splits: {list(ds.keys())}")

    # Discover column names
    corpus_split = next(s for s in ds.keys() if "corpus" in s.lower())
    query_split = next(s for s in ds.keys() if "quer" in s.lower())

    corpus = ds[corpus_split]
    queries = ds[query_split]
    print(f"  Corpus: {len(corpus)} pages")
    print(f"  Queries: {len(queries)} queries")
    print(f"  Corpus columns: {corpus.column_names}")
    print(f"  Query columns: {queries.column_names}")

    # Find image and id columns
    corpus_cols = corpus.column_names
    id_col = next(c for c in corpus_cols if "id" in c.lower())
    img_col = next(c for c in corpus_cols if c in ("image", "img", "page_image"))
    q_id_col = next(c for c in queries.column_names if "id" in c.lower())
    q_text_col = next(c for c in queries.column_names
                      if c in ("query", "text", "question"))

    print(f"  Using: id={id_col}, image={img_col}, q_id={q_id_col}, q_text={q_text_col}")

    # --- Extract data ---
    corpus_ids = [str(row[id_col]) for row in corpus]
    corpus_images = [row[img_col] for row in corpus]

    # --- Check for checkpoint (partial progress) ---
    checkpoint_path = os.path.join(out_dir, "checkpoint.json")
    start_idx = 0
    page_embeddings = []

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        start_idx = ckpt["pages_done"]
        # Load partial embeddings
        partial_path = os.path.join(out_dir, "pages_partial.npz")
        if os.path.exists(partial_path):
            data = np.load(partial_path, allow_pickle=True)
            page_embeddings = [data[f"page_{i}"] for i in range(len(data.files))]
            print(f"  Resuming from checkpoint: {start_idx}/{len(corpus_images)} pages done")

    # --- Embed pages ---
    if start_idx < len(corpus_images):
        print(f"\nEmbedding pages {start_idx}..{len(corpus_images)}...")
        t_embed = time.time()

        remaining_images = corpus_images[start_idx:]
        total = len(remaining_images)

        for i in range(0, total, BATCH_SIZE):
            batch_imgs = remaining_images[i:i + BATCH_SIZE]
            batch = processor.process_images(batch_imgs)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                embeddings = model(**batch)

            for emb in embeddings:
                page_embeddings.append(emb.cpu().float().numpy())

            del batch, embeddings
            if device == "cuda":
                torch.cuda.empty_cache()

            done = start_idx + min(i + BATCH_SIZE, total)
            total_pages = len(corpus_images)

            if (done - start_idx) % 10 == 0 or done == total_pages:
                elapsed = time.time() - t_embed
                pages_this_run = done - start_idx
                rate = elapsed / max(pages_this_run, 1)
                remaining = total_pages - done
                eta = rate * remaining
                print(f"  {done}/{total_pages} pages  "
                      f"({rate:.1f}s/page, ~{eta/60:.0f}min remaining)")

            # Checkpoint every N pages
            if (done - start_idx) % CHECKPOINT_EVERY == 0 and done < total_pages:
                save_embeddings(page_embeddings, [], os.path.join(out_dir, "pages_partial.npz"))
                with open(checkpoint_path, "w") as f:
                    json.dump({"pages_done": done, "total": total_pages}, f)
                print(f"  -- checkpoint at {done} pages --")

        embed_time = time.time() - t_embed
        print(f"\nEmbedding done in {embed_time/60:.1f} min "
              f"({embed_time/len(corpus_images):.1f} sec/page)")
    else:
        embed_time = 0

    # --- Save page embeddings ---
    page_meta = [{"corpus_id": cid, "index": i} for i, cid in enumerate(corpus_ids)]
    save_embeddings(page_embeddings, page_meta, os.path.join(out_dir, "pages.npz"))

    # --- Embed queries ---
    query_ids = [str(row[q_id_col]) for row in queries]
    query_texts = [row[q_text_col] for row in queries]

    print(f"\nEmbedding {len(query_texts)} queries...")
    query_embeddings = embed_queries(model, processor, query_texts, device)
    query_meta = [{"query_id": qid, "text": qt, "index": i}
                  for i, (qid, qt) in enumerate(zip(query_ids, query_texts))]
    save_embeddings(query_embeddings, query_meta, os.path.join(out_dir, "queries.npz"))

    # --- Prune ---
    print(f"\nPruning (keep top {KEEP_FRACTION:.0%} patches)...")
    pruned = prune_l2_topk(page_embeddings, keep_fraction=KEEP_FRACTION)
    raw_patches = sum(e.shape[0] for e in page_embeddings)
    pruned_patches = sum(e.shape[0] for e in pruned)
    print(f"  {raw_patches} -> {pruned_patches} patches "
          f"({pruned_patches/raw_patches:.1%})")
    save_embeddings(pruned, page_meta, os.path.join(out_dir, "pages_pruned.npz"))

    # --- Save qrels ---
    qrels_split = next((s for s in ds.keys() if "qrel" in s.lower()), None)
    if qrels_split:
        qrels_data = ds[qrels_split]
        qrels_list = []
        for row in qrels_data:
            entry = {}
            for k, v in dict(row).items():
                entry[k] = v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v)
            qrels_list.append(entry)
        with open(os.path.join(out_dir, "qrels.json"), "w") as f:
            json.dump(qrels_list, f)
        print(f"  Saved {len(qrels_list)} qrels")

    # --- Save results ---
    results = {
        "dataset": dataset_key,
        "model": MODEL_NAME,
        "prune_fraction": KEEP_FRACTION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_pages": len(corpus_images),
        "n_queries": len(query_texts),
        "embed_time_sec": embed_time,
        "sec_per_page": embed_time / max(len(corpus_images), 1),
        "raw_patches": raw_patches,
        "pruned_patches": pruned_patches,
        "compression_ratio": raw_patches / pruned_patches,
        "dim": page_embeddings[0].shape[1] if page_embeddings else None,
    }
    with open(done_marker, "w") as f:
        json.dump(results, f, indent=2)

    # Clean up checkpoint
    for f_name in ["pages_partial.npz", "pages_partial_meta.json", "checkpoint.json"]:
        p = os.path.join(out_dir, f_name)
        if os.path.exists(p):
            os.remove(p)

    print(f"\nDone: {dataset_key}")
    print(json.dumps(results, indent=2))


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "hr"

    if target == "all":
        keys = list(DATASETS.keys())
    elif target in DATASETS:
        keys = [target]
    else:
        print(f"Unknown dataset: {target}")
        print(f"Options: {', '.join(DATASETS.keys())}, all")
        sys.exit(1)

    print(f"Datasets to process: {keys}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    model, processor, device = load_model()

    for key in keys:
        process_dataset(key, model, processor, device)

    print(f"\n{'=' * 60}")
    print("All done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
