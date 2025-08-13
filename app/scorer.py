from typing import Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model: SentenceTransformer | None = None

def _ensure_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # small & fast
    return _model

def score_essay(text: str, prompt: Optional[str]) -> tuple[float, dict]:
    """
    Returns a 0-100 score emphasizing semantic alignment to the prompt.
    If no prompt is provided, fall back to a simple 'richness' proxy.
    """
    model = _ensure_model()

    if prompt and prompt.strip():
        embs = model.encode([prompt, text], normalize_embeddings=True)
        sim = cosine_similarity([embs[0]], [embs[1]])[0][0]  # -1..1
        raw = max(0.0, sim) ** 0.5 * 100.0
        details = {"model": "all-MiniLM-L6-v2", "strategy": "prompt_similarity", "sim": float(sim)}
        return float(raw), details

    # Fallback when no prompt: basic richness proxy (length + uniqueness)
    tokens = text.split()
    uniq = len(set(tokens))
    richness = uniq / max(1, len(tokens))
    raw = min(100.0, (len(tokens) / 300.0) * 60.0 + richness * 40.0)
    details = {"model": "all-MiniLM-L6-v2", "strategy": "richness_proxy"}
    return float(raw), details
