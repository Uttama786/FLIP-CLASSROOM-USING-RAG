"""Utilities for loading the sentence-transformer model with resilience."""

import os
import time

EMBEDDING_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_EMBEDDING_MODEL = None


def _is_transient_model_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    transient_markers = (
        "forcibly closed",
        "connection reset",
        "temporarily unavailable",
        "timed out",
        "timeout",
        "cannot send a request, as the client has been closed",
        "network is unreachable",
        "connection aborted",
    )
    return any(marker in msg for marker in transient_markers)


def get_embedding_model():
    """Load and cache a SentenceTransformer model with retry + local fallback."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    try:
        from flipped_app.torch_patch import patch_torch_compat
        patch_torch_compat()
    except Exception:
        pass

    from sentence_transformers import SentenceTransformer

    model_candidates = []
    if EMBEDDING_MODEL_NAME:
        model_candidates.append(EMBEDDING_MODEL_NAME)
    if not EMBEDDING_MODEL_NAME.startswith("sentence-transformers/"):
        model_candidates.append(f"sentence-transformers/{EMBEDDING_MODEL_NAME}")

    # 1. First try loading from local cache (instant, avoids network checks and HF rate limits)
    for model_name in model_candidates:
        try:
            _EMBEDDING_MODEL = SentenceTransformer(model_name, local_files_only=True)
            return _EMBEDDING_MODEL
        except Exception:
            pass

    # 2. Download from Hugging Face if not present locally
    last_error = None
    for model_name in model_candidates:
        for attempt in range(1, 4):
            try:
                _EMBEDDING_MODEL = SentenceTransformer(model_name)
                return _EMBEDDING_MODEL
            except Exception as exc:
                last_error = exc
                if attempt < 3 and _is_transient_model_error(exc):
                    sleep_s = attempt
                    print(
                        f"[Embedding] Model load failed ({model_name}), retrying in {sleep_s}s "
                        f"[{attempt}/3]: {exc}"
                    )
                    time.sleep(sleep_s)
                    continue
                break

    raise RuntimeError(
        "Unable to load embedding model from Hugging Face or local cache. "
        "Check network access to huggingface.co, or pre-download the model and retry."
    ) from last_error
