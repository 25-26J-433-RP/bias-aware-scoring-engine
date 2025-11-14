import re

def clean_sinhala(text: str) -> str:
    # remove English, numbers, extra punctuation
    text = re.sub(r'[a-zA-Z0-9]', '', text)
    text = re.sub(r'[“”"\'.,!?;:()\[\]{}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_features(text: str):
    words = text.split()
    word_count = len(words)
    unique_words = len(set(words))
    avg_word_len = sum(len(w) for w in words) / word_count if word_count else 0
    return word_count, unique_words, avg_word_len


def baseline_sinhala_score(text: str) -> tuple[float, dict]:
    """
    Returns a 0–100 Sinhala baseline score using a simple heuristic.
    """

    cleaned = clean_sinhala(text)
    wc, uw, avg_len = extract_features(cleaned)

    # Heuristic weights
    score = (
        (wc / 250 * 40) +        # essay length weight = 40%
        (uw / 150 * 40) +        # vocabulary richness = 40%
        (avg_len / 6 * 20)       # avg word length = 20%
    )

    final_score = min(round(score, 2), 100)

    details = {
        "strategy": "sinhala_rule_based",
        "word_count": wc,
        "unique_words": uw,
        "avg_word_length": round(avg_len, 2)
    }

    return final_score, details
