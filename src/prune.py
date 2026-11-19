"""DocPruner: reduce patch count per page by keeping highest L2-norm patches."""

import numpy as np


def prune_l2_topk(embeddings: list[np.ndarray],
                  keep_fraction: float = 0.5) -> list[np.ndarray]:
    """Keep top patches by L2 norm per page.

    Args:
        embeddings: list of (n_patches, dim) arrays, one per page
        keep_fraction: fraction of patches to keep (0.5 = 2x compression)

    Returns:
        list of (n_kept, dim) arrays with spatial order preserved
    """
    pruned = []
    for emb in embeddings:
        norms = np.linalg.norm(emb, axis=1)
        k = max(1, int(len(norms) * keep_fraction))
        top_idx = np.argpartition(norms, -k)[-k:]
        top_idx.sort()  # keep spatial order
        pruned.append(emb[top_idx])
    return pruned


def compression_report(original: list[np.ndarray],
                       pruned: list[np.ndarray]) -> dict:
    """Measure compression stats."""
    orig_patches = sum(e.shape[0] for e in original)
    pruned_patches = sum(e.shape[0] for e in pruned)
    dim = original[0].shape[1]
    bytes_per_float = 4

    orig_bytes = orig_patches * dim * bytes_per_float
    pruned_bytes = pruned_patches * dim * bytes_per_float

    return {
        "pages": len(original),
        "original_patches": orig_patches,
        "pruned_patches": pruned_patches,
        "original_mb": orig_bytes / 1e6,
        "pruned_mb": pruned_bytes / 1e6,
        "compression_ratio": orig_patches / pruned_patches,
        "patches_per_page_before": orig_patches / len(original),
        "patches_per_page_after": pruned_patches / len(pruned),
    }


# --- Self-check ---

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Simulate 10 pages with 1024 patches each, dim 128
    fake_embeddings = [rng.standard_normal((1024, 128)).astype(np.float32)
                       for _ in range(10)]

    pruned = prune_l2_topk(fake_embeddings, keep_fraction=0.5)

    assert len(pruned) == 10
    assert all(e.shape == (512, 128) for e in pruned), \
        f"Expected (512, 128), got {pruned[0].shape}"

    report = compression_report(fake_embeddings, pruned)
    assert abs(report["compression_ratio"] - 2.0) < 0.01

    print(f"Compression: {report['compression_ratio']:.2f}x")
    print(f"  {report['original_patches']} -> {report['pruned_patches']} patches")
    print(f"  {report['original_mb']:.1f} MB -> {report['pruned_mb']:.1f} MB")
    print("\nOK prune.py self-check passed")
