# app/grade_detector.py

"""
Grade detection module for inferring the student grade level from essay text.

This module analyzes linguistic features to estimate the grade level (3-8) of a Sinhala essay.
"""

import re
from typing import Optional


def detect_grade(text: str) -> int:
    """
    Detect the approximate grade level (3-8) of a Sinhala essay based on linguistic features.
    
    Features analyzed:
    - Text length and sentence complexity
    - Vocabulary sophistication (estimating by rare character patterns)
    - Sentence structure diversity
    - Common grade-level markers
    
    Args:
        text: The Sinhala essay text
        
    Returns:
        Estimated grade level (3-8). Default is 5 (middle grade).
    """
    if not text or len(text.strip()) == 0:
        return 3  # Default to lowest grade for empty text
    
    # Calculate various linguistic features
    features = {
        "length_score": _analyze_length(text),
        "complexity_score": _analyze_complexity(text),
        "vocabulary_score": _analyze_vocabulary(text),
    }
    
    # Weighted average of features
    grade_estimate = (
        features["length_score"] * 0.3 +
        features["complexity_score"] * 0.4 +
        features["vocabulary_score"] * 0.3
    )
    
    # Normalize to grade range 3-8
    grade = int(round(grade_estimate))
    grade = max(3, min(8, grade))  # Clamp to valid range
    
    return grade


def _analyze_length(text: str) -> float:
    """
    Analyze text length as an indicator of grade level.
    Longer, more detailed essays typically indicate higher grades.
    """
    word_count = len(text.split())
    sentence_count = len(re.split(r'[।\.!\?]+', text.strip())) - 1
    
    if sentence_count == 0:
        sentence_count = 1
    
    avg_words_per_sentence = word_count / sentence_count
    
    # Grade 3: ~15-30 words, avg 3-5 words/sentence
    # Grade 4: ~30-80 words, avg 5-10 words/sentence
    # Grade 5: ~80-150 words, avg 10-15 words/sentence  
    # Grade 6+: ~150-250 words, avg 15+ words/sentence
    # Grade 8: ~250+ words, avg 15+ words/sentence
    
    # Normalize length to 3-8 scale
    if word_count < 15:
        return 3.0
    elif word_count < 30:
        return 4.0
    elif word_count < 80:
        return 5.0
    elif word_count < 150:
        return 6.0
    elif word_count < 250:
        return 7.0
    else:
        return 8.0


def _analyze_complexity(text: str) -> float:
    """
    Analyze sentence structure complexity.
    More complex sentences with multiple clauses indicate higher grades.
    """
    sentences = re.split(r'[।\.!\?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 3.0
    
    complexity_scores = []
    for sentence in sentences:
        # Count conjunctions and complex structures
        # Common Sinhala conjunctions: ඉතා, එවිට, කෙසේ නම්, ද, ගෙන, ගත්, විට, හෝ
        conjunctions = len(re.findall(r'(ඉතා|එවිට|ගෙන|විට|හෝ|ද|ගත්|ගැන|ගේ)', sentence))
        
        # Count clauses (roughly by commas and complex markers)
        commas = sentence.count('،') + sentence.count(',')
        parentheses = sentence.count('(') + sentence.count('(')
        
        clause_count = max(1, conjunctions + commas // 2)
        word_in_sentence = len(sentence.split())
        
        # Normalize: more clauses and longer sentences = higher complexity
        if word_in_sentence < 5:
            score = 3.0
        elif word_in_sentence < 10:
            score = 4.0 + (clause_count * 0.3)
        elif word_in_sentence < 20:
            score = 5.0 + (clause_count * 0.4)
        else:
            score = 6.0 + (clause_count * 0.5)
        
        complexity_scores.append(min(8.0, score))
    
    # Return average complexity
    return sum(complexity_scores) / len(complexity_scores)


def _analyze_vocabulary(text: str) -> float:
    """
    Analyze vocabulary sophistication.
    Uses word length and character patterns as proxies for vocabulary difficulty.
    """
    words = text.split()
    
    if not words:
        return 3.0
    
    # Calculate average word length and letter count
    word_lengths = [len(word) for word in words]
    avg_word_length = sum(word_lengths) / len(word_lengths)
    
    # Count words with special prefixes/suffixes (indicating derived words, higher sophistication)
    # Common Sinhala prefixes: අ-, නැ-, නොවා-, ප්‍රධාන-
    # Common Sinhala suffixes: -කරු, -කරණ, -වන, -නම්, -ගේ
    sophisticated_patterns = len(re.findall(r'(කරු|කරණ|වන|නම්|ගේ|හා|ලා|ම|ය|ක)', text))
    
    # Rare/complex characters (not commonly used in lower grades)
    complex_chars = len(re.findall(r'[^\u0D80-\u0DF0a-zA-Z0-9 ।,\.!?\-()]', text))
    
    sophistication_score = sophisticated_patterns + (complex_chars * 0.1)
    
    # Map features to grade scale
    # Grade 3: avg 3-4 chars/word, 0-5 sophisticated patterns
    # Grade 4-5: avg 4-5 chars/word, 5-20 sophisticated patterns
    # Grade 6+: avg 5+ chars/word, 20+ sophisticated patterns
    
    if avg_word_length < 3.5:
        base_score = 3.0
    elif avg_word_length < 4.5:
        base_score = 4.0
    elif avg_word_length < 5.5:
        base_score = 5.0
    elif avg_word_length < 6.5:
        base_score = 6.0
    else:
        base_score = 7.0
    
    # Adjust based on sophistication
    if sophistication_score < 5:
        vocab_score = base_score - 0.5
    elif sophistication_score < 15:
        vocab_score = base_score + 0.2
    elif sophistication_score < 25:
        vocab_score = base_score + 0.5
    else:
        vocab_score = base_score + 1.0
    
    return min(8.0, max(3.0, vocab_score))


def infer_grade_from_text(text: Optional[str], provided_grade: Optional[int]) -> int:
    """
    Intelligently handle grade detection.
    
    If a grade is explicitly provided, use it (trust the user).
    Otherwise, detect it from the text.
    
    Args:
        text: The essay text
        provided_grade: User-provided grade (can be None)
        
    Returns:
        Final grade to use for scoring (3-8)
    """
    if provided_grade is not None:
        # Validate and use provided grade
        provided_grade = max(3, min(8, int(provided_grade)))
        return provided_grade
    
    # Detect grade from text
    if text and text.strip():
        return detect_grade(text)
    
    # Default fallback
    return 5
