from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity

# Lazy import of SentenceTransformer to avoid forcing heavy deps at container build time.
_model = None

def _ensure_model():
    """Return a SentenceTransformer instance or None if the package isn't installed."""
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return _model
    except Exception:
        # sentence-transformers (and its torch dependency) is not available.
        _model = None
        return None

def score_essay(text: str, prompt: Optional[str]) -> tuple[float, dict]:
    """
    Returns a 0-100 score emphasizing semantic alignment to the prompt.
    If no prompt is provided, fall back to a simple 'richness' proxy.
    """
    model = _ensure_model()

    if prompt and prompt.strip():
        # If sentence-transformers is available, use it for embeddings (preferred).
        if model is not None:
            embs = model.encode([prompt, text], normalize_embeddings=True)
            sim = cosine_similarity([embs[0]], [embs[1]])[0][0]  # -1..1
            raw = max(0.0, sim) ** 0.5 * 100.0
            details = {"model": "all-MiniLM-L6-v2", "strategy": "prompt_similarity", "sim": float(sim)}
            return float(raw), details

        # Fallback: use a lightweight TF-IDF + cosine similarity when sentence-transformers isn't installed.
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer().fit_transform([prompt, text])
        sims = cosine_similarity(vec[0], vec[1])[0][0]
        raw = max(0.0, float(sims)) * 100.0
        details = {"model": "tfidf-fallback", "strategy": "prompt_similarity", "sim": float(sims)}
        return float(raw), details

    # Fallback when no prompt: basic richness proxy (length + uniqueness)
    tokens = text.split()
    uniq = len(set(tokens))
    richness = uniq / max(1, len(tokens))
    raw = min(100.0, (len(tokens) / 300.0) * 60.0 + richness * 40.0)
    details = {"model": "all-MiniLM-L6-v2", "strategy": "richness_proxy"}
    return float(raw), details
