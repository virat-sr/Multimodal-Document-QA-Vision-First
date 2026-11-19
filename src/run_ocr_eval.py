"""Run OCR baseline evaluation on ViDoRe v3 dataset and compare to vision RAG."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from datasets import load_dataset

from src.ocr_baseline import OCRPipeline


def run_ocr_evaluation(dataset_key="hr"):
    DATASET_MAP = {
        "hr": "vidore/vidore_v3_hr",
        "computer_science": "vidore/vidore_v3_computer_science",
        "finance_en": "vidore/vidore_v3_finance_en",
    }

    dataset_name = DATASET_MAP[dataset_key]
    print(f"{'=' * 60}")
    print(f"OCR Baseline Evaluation: {dataset_key}")
    print(f"{'=' * 60}")

    # --- Load dataset ---
    print(f"\nLoading {dataset_name}...")
    corpus = load_dataset(dataset_name, "corpus", trust_remote_code=True)["test"]
    queries = load_dataset(dataset_name, "queries", trust_remote_code=True)["test"]
    qrels_raw = load_dataset(dataset_name, "qrels", trust_remote_code=True)["test"]

    corpus_ids = [str(row["corpus_id"]) for row in corpus]
    corpus_images = [row["image"] for row in corpus]
    query_ids = [str(row["query_id"]) for row in queries]
    query_texts = [row["query"] for row in queries]

    print(f"  {len(corpus_images)} pages, {len(query_texts)} queries")

    # --- Build qrels dict ---
    qrels_dict = {}
    for row in qrels_raw:
        qid = str(row["query_id"])
        cid = str(row["corpus_id"])
        score = int(row["score"])
        qrels_dict.setdefault(qid, {})[cid] = score

    # --- Run OCR pipeline ---
    pipeline = OCRPipeline()
    t0 = time.time()
    pipeline.index_pages(corpus_images, corpus_ids)
    index_time = time.time() - t0

    # --- Evaluate ---
    print(f"\nRunning retrieval for {len(query_texts)} queries...")
    import pytrec_eval

    t_eval = time.time()
    run_dict = pipeline.batch_search(query_texts, query_ids, top_k=10)
    eval_time = time.time() - t_eval

    # Filter to queries that have qrels
    run_dict = {qid: scores for qid, scores in run_dict.items() if qid in qrels_dict}
    print(f"  {len(run_dict)} queries with qrels evaluated")

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_dict, {"ndcg_cut_5", "ndcg_cut_10", "recall_5", "recall_10", "map"}
    )
    eval_results = evaluator.evaluate(run_dict)

    metrics = {}
    for m in ["ndcg_cut_5", "ndcg_cut_10", "recall_5", "recall_10", "map"]:
        vals = [r[m] for r in eval_results.values()]
        metrics[m] = sum(vals) / len(vals) if vals else 0

    # Latency
    latency = pipeline.measure_latency(query_texts, n_queries=100)
    metrics["retrieval_p50_ms"] = latency["p50_ms"]
    metrics["retrieval_p95_ms"] = latency["p95_ms"]

    # --- Print results ---
    print(f"\n{'=' * 50}")
    print(f"  {dataset_key} -- OCR RAG Baseline")
    print(f"{'=' * 50}")
    print(f"  NDCG@5:     {metrics['ndcg_cut_5']:.4f}")
    print(f"  NDCG@10:    {metrics['ndcg_cut_10']:.4f}")
    print(f"  Recall@5:   {metrics['recall_5']:.4f}")
    print(f"  Recall@10:  {metrics['recall_10']:.4f}")
    print(f"  MAP:        {metrics['map']:.4f}")
    print(f"  p95 latency: {metrics['retrieval_p95_ms']:.0f}ms")
    print(f"{'=' * 50}")

    # --- Compare to vision RAG ---
    vision_path = f"eval/vidore_vision_results.json"
    if os.path.exists(vision_path):
        with open(vision_path) as f:
            vision_results = json.load(f)
        if dataset_key in vision_results:
            v = vision_results[dataset_key]
            print(f"\n  COMPARISON: Vision RAG vs OCR RAG")
            print(f"  {'Metric':<15} {'Vision':>10} {'OCR':>10} {'Diff':>10}")
            print(f"  {'-'*45}")
            for m_name, m_label in [
                ("ndcg_cut_5", "NDCG@5"),
                ("recall_5", "Recall@5"),
                ("recall_10", "Recall@10"),
                ("map", "MAP"),
            ]:
                v_val = v.get(m_name, 0)
                o_val = metrics[m_name]
                diff = v_val - o_val
                sign = "+" if diff > 0 else ""
                print(f"  {m_label:<15} {v_val:>10.4f} {o_val:>10.4f} {sign}{diff:>9.4f}")

    # --- Save results ---
    results = {
        "dataset": dataset_key,
        "pipeline": "ocr_baseline",
        "model": "all-MiniLM-L6-v2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_pages": len(corpus_images),
        "n_queries": len(query_texts),
        "n_chunks": len(pipeline.chunks),
        "index_time_sec": index_time,
        **metrics,
    }

    os.makedirs("eval", exist_ok=True)
    out_path = f"eval/ocr_baseline_{dataset_key}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    return results


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "hr"
    run_ocr_evaluation(dataset)
