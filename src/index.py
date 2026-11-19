"""MaxSim retrieval index — pure numpy, no vector DB."""

import json
import time

import numpy as np


class MaxSimIndex:
    """In-memory MaxSim late-interaction retrieval.

    Stores multi-vector page embeddings and scores queries using
    the MaxSim operator: for each query token, find the max similarity
    across all patches of each page, then sum.
    """

    def __init__(self, page_embeddings: list[np.ndarray], metadata: list[dict]):
        """
        Args:
            page_embeddings: list of (n_patches, dim) arrays, one per page
            metadata: list of dicts with at least 'corpus_id' per page
        """
        self.page_embeddings = page_embeddings
        self.metadata = metadata
        self.n_pages = len(page_embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Retrieve top-k pages for a single query.

        Args:
            query_embedding: (n_tokens, dim) array
            top_k: number of results

        Returns:
            list of {corpus_id, score, rank, ...metadata} sorted by score desc
        """
        scores = self._maxsim(query_embedding)
        top_k = min(top_k, self.n_pages)
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = []
        for rank, idx in enumerate(top_idx):
            results.append({
                **self.metadata[idx],
                "score": float(scores[idx]),
                "rank": rank,
            })
        return results

    def batch_search(self, query_embeddings: list[np.ndarray],
                     query_ids: list[str],
                     top_k: int = 10) -> dict[str, dict[str, float]]:
        """Batch search for pytrec_eval compatibility.

        Returns:
            {query_id: {corpus_id: score}} for top-k results per query
        """
        run = {}
        for q_emb, qid in zip(query_embeddings, query_ids):
            results = self.search(q_emb, top_k=top_k)
            run[qid] = {r["corpus_id"]: r["score"] for r in results}
        return run

    def measure_latency(self, query_embeddings: list[np.ndarray],
                        n_queries: int = 100) -> dict:
        """Measure retrieval latency stats."""
        n = min(n_queries, len(query_embeddings))
        latencies = []

        for q_emb in query_embeddings[:n]:
            t0 = time.perf_counter()
            self._maxsim(q_emb)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

        lat = np.array(latencies)
        return {
            "n_queries": n,
            "n_pages": self.n_pages,
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)),
            "mean_ms": float(lat.mean()),
        }

    def _maxsim(self, query_embedding: np.ndarray) -> np.ndarray:
        """Core MaxSim scoring.

        For each query token, find max similarity against every patch
        of every page, then sum across query tokens.

        query_embedding: (q_tokens, dim)
        Returns: (n_pages,) scores
        """
        scores = np.zeros(self.n_pages)
        for i, page_emb in enumerate(self.page_embeddings):
            # (q_tokens, dim) @ (dim, n_patches) -> (q_tokens, n_patches)
            sims = query_embedding @ page_emb.T
            # Max per query token, then sum
            scores[i] = sims.max(axis=1).sum()
        return scores


# --- Loading helpers ---

def load_embeddings(npz_path: str) -> tuple[list[np.ndarray], list[dict]]:
    """Load embeddings from .npz + metadata JSON sidecar."""
    data = np.load(npz_path, allow_pickle=True)
    embeddings = [data[f"page_{i}"] for i in range(len(data.files))]

    meta_path = npz_path.replace(".npz", "_meta.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    return embeddings, metadata


def build_index(embeddings_dir: str, dataset_key: str,
                pruned: bool = True) -> MaxSimIndex:
    """Build a MaxSimIndex from saved embeddings.

    Args:
        embeddings_dir: path to embeddings/ directory
        dataset_key: e.g. 'finance_en'
        pruned: use pruned embeddings (default True)
    """
    suffix = "pages_pruned" if pruned else "pages"
    npz_path = f"{embeddings_dir}/{dataset_key}/{suffix}.npz"

    embeddings, metadata = load_embeddings(npz_path)
    print(f"Index: {len(embeddings)} pages, "
          f"~{sum(e.shape[0] for e in embeddings)} patches, "
          f"dim={embeddings[0].shape[1]}")

    return MaxSimIndex(embeddings, metadata)


# --- Self-check ---

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Simulate 50 pages with 512 patches each (pruned), dim 128
    fake_pages = [rng.standard_normal((512, 128)).astype(np.float32)
                  for _ in range(50)]
    fake_meta = [{"corpus_id": f"page_{i}", "index": i} for i in range(50)]

    index = MaxSimIndex(fake_pages, fake_meta)

    # Simulate a query with 32 tokens
    fake_query = rng.standard_normal((32, 128)).astype(np.float32)

    results = index.search(fake_query, top_k=5)
    assert len(results) == 5
    assert results[0]["score"] >= results[1]["score"], "Results should be sorted by score"
    assert "corpus_id" in results[0]
    assert "rank" in results[0]

    print("Top-5 results:")
    for r in results:
        print(f"  {r['corpus_id']}: score={r['score']:.2f}")

    # Latency
    fake_queries = [rng.standard_normal((32, 128)).astype(np.float32) for _ in range(20)]
    lat = index.measure_latency(fake_queries, n_queries=20)
    print(f"\nLatency (50 pages, 512 patches):")
    print(f"  p50: {lat['p50_ms']:.1f}ms")
    print(f"  p95: {lat['p95_ms']:.1f}ms")

    # Batch search
    fake_qids = [f"q_{i}" for i in range(5)]
    run = index.batch_search(fake_queries[:5], fake_qids, top_k=3)
    assert len(run) == 5
    assert all(len(v) == 3 for v in run.values())

    print("\nOK index.py self-check passed")
