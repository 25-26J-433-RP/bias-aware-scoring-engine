# app/rubric_evaluator.py

"""
Hybrid Rubric Evaluator — Rule-Based Component
================================================
This module implements the HARD RULES for the Sinhala essay grading rubric:

Architecture:  ML Model (General Quality) + Hard Rules (Specific Rubric)
                ↓                            ↓
          richness_5, org_6           technical_3 adjustments
          (from XLM-R)               (from regex + heuristics)

Marking Scheme (02):
  • සාරාංශය                              — ලකුණු 05
  • සංවිධානය, නිර්මාණශීලිභාවය හා ප්‍රත්‍යාර්ථය  — ලකුණු 06
  • ගීල්පීය දක්ෂතා සහ වේද ලිවීම          — ලකුණු 03
  • මුළු ලකුණු 14

Sections:
  1. Punctuation & Layout Analysis  (6 Rules → affects technical_3)
  2. Heuristic Grammar Checks       (word sep, sentence len, verb markers → affects technical_3)
  3. Topic/Theme Relevance           (keyword freq + cosine similarity → affects richness_5)
  4. Word Count Penalty              (min 150 words → affects richness_5)
"""

import re
import torch
from typing import Optional, List, Dict, Any


class RubricEvaluator:
    """
    Rule-based evaluator that augments ML predictions with hard rubric checks.
    
    The ML model (XLM-R) provides general quality scores, but the marking scheme
    requires specific measurable criteria that rules handle more accurately.
    """

    def __init__(self):
        # ═══════════════════════════════════════════════════════════════
        # PUNCTUATION & LAYOUT RULES (6 Rules from Marking Scheme)
        # ═══════════════════════════════════════════════════════════════
        
        # Rule 1: Space before punctuation marks (., : ,)
        # Wrong: "වචනය ." → Correct: "වචනය."
        self.space_before_punct = re.compile(r'\s+[.,:;]')
        
        # Rule 2: Space inside quotation marks
        # Wrong: "" වචනය "" → Correct: ""වචනය""
        self.space_inside_quotes = re.compile(r'"\s+|\s+"')
        
        # Rule 3: Space inside brackets
        # Wrong: "( වචනය )" → Correct: "(වචනය)"
        self.space_inside_brackets = re.compile(r'\(\s+|\s+\)')
        
        # Rule 4: Missing full stop (Sinhala purnawishramaya: .)
        # Sentences > 15 words without ending punctuation
        self.sentence_ender = re.compile(r'[.!?。।]')
        
        # Rule 5: Ellipsis must be exactly 3 dots
        # Wrong: ".." or "...." → Correct: "..."
        self.ellipsis_err = re.compile(r'(?<!\.)\.\.(?!\.)(?!$)|(?<!\.)\\.{4,}')
        
        # Rule 6: English abbreviation dots mixed with Sinhala
        # e.g., "U.N.O" mixed into Sinhala text without proper spacing
        self.mixed_abbrev = re.compile(r'\b[A-Za-z]\.[A-Za-z]\b(?!\.)')

        # ═══════════════════════════════════════════════════════════════
        # HEURISTIC GRAMMAR PATTERNS (Sinhala-specific)
        # ═══════════════════════════════════════════════════════════════
        
        # Common Sinhala verb endings (present, past, future tense markers)
        # These help detect if sentences have proper verb structure
        self.sinhala_verb_markers = re.compile(
            r'(යි|ය|ේය|ුය|ීය|ුනි|ුණි'        # Past tense markers
            r'|නවා|නු ලැබේ|කරයි|කළේය'          # Present/Active voice
            r'|ඇත|ඇති|නැත|වේ|විය'             # Existential / auxiliary
            r'|දේ|ගනී|ලැබේ|යුතුය|හැකිය'       # Modal / potential
            r'|බැරිය|ගියේය|ආවේය|වුණි'          # Additional common endings
            r'|කරති|කරන|සිටී|සිටියේය'          # Plural / continuous
            r'|ඕනෑ|නිසා|නිසයි|ආදියයි'         # Common essay endings
            r'|වෙනවා|හැදෙනවා|සෑදෙනවා'         # Colloquial present
            r'|තැන්ය|ලෙඩය|වැඩිය|එපා)$'        # Predicative endings
        )
        
        # Fragment markers: Suffixes that imply the sentence should continue
        # If the sentence ends with these + a full stop, it's a "Grammar Fragment"
        self.sinhala_fragment_enders = re.compile(
            r'(මින්|දී|මේදී|තැන්දී|විටදී|ට|මට|වීමට|සඳහා|ලා|බව|බවත්)[.!?।\s]*$'
        )

        # Common Sinhala sentence connectors (subordinating conjunctions)
        self.sinhala_connectors = re.compile(
            r'(නමුත්|එහෙත්|එනමුත්|එසේම|තවද'
            r'|මීට අමතරව|ඒ අනුව|එබැවින්'
            r'|එමෙන්ම|අනෙක් අතට|පළමුව|දෙවනුව'
            r'|එවිට|ඉන්පසු|අවසානයේ|මෙහිදී'
            r'|ඒ නිසා|මන්ද|සමහරවිට'
            r'|ඒත්|නිසා|නිසයි|ඒක|ඒවායේ)'      # Added simpler connectors
        )
        
        # Word separation patterns — Sinhala words joined without spaces
        # Extremely long tokens (>25 chars) are likely concatenated words
        self.max_word_length = 25

        # Sinhala topic filler words that should be ignored for keyword scoring
        # to prevent common words like "My" (මගේ) from inflating relevance scores.
        self.sinhala_topic_stopwords = {
            "මගේ", "අපේ", "මම", "අපි", "ඔබගේ", "ගැන", "පිළිබඳ", "පිළිබඳව",
            "රචනය", "රචනයක්", "මාතෘකාව", "විස්තරය", "වැදගත්කම",
            "ප්‍රයෝජන", "ප්‍රයෝජනය", "ආදිය", "සහ", "හා", "පිළිබද", "පිළිබදව"
        }

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: PUNCTUATION & LAYOUT ANALYSIS
    # Directly affects: technical_3 (out of 3)
    # ═══════════════════════════════════════════════════════════════
    
    def analyze_technical(self, text: str) -> Dict[str, Any]:
        """
        Evaluate the essay against the 6 punctuation/layout rules AND
        heuristic grammar checks.
        
        Returns:
            {
                "penalty": float (0..2.0, deducted from technical_3),
                "violations": list of human-readable violation descriptions,
                "grammar_issues": list of grammar-related issues
            }
        """
        violations = []
        grammar_issues = []
        
        # ─── 6 Punctuation & Layout Rules ───
        
        # Rule 1: Space before punctuation
        matches_r1 = self.space_before_punct.findall(text)
        if matches_r1:
            violations.append(f"Rule 1: Space before punctuation (.,: ;) — {len(matches_r1)} instance(s)")
        
        # Rule 2: Space inside quotation marks
        if self.space_inside_quotes.search(text):
            violations.append("Rule 2: Space inside quotation marks")
        
        # Rule 3: Space inside brackets
        if self.space_inside_brackets.search(text):
            violations.append("Rule 3: Space inside brackets")
        
        # Rule 4: Missing full stop / sentence-ending punctuation
        # Join continuation lines first before checking
        joined_text = self._join_continuation_lines(text)
        missing_fullstop = self._check_missing_fullstop(joined_text)
        if missing_fullstop:
            violations.append(f"Rule 4: Missing sentence-ending punctuation — {missing_fullstop} sentence(s)")
        
        # Rule 5: Ellipsis not exactly 3 dots
        if self.ellipsis_err.search(text):
            violations.append("Rule 5: Ellipsis should be exactly 3 dots (...)")
        
        # Rule 6: Mixed English abbreviation dots
        if self.mixed_abbrev.search(text):
            violations.append("Rule 6: English abbreviation dots mixed with Sinhala text")
        
        # ─── Heuristic Grammar Checks ───
        grammar_issues = self._analyze_grammar(text)
        
        # ─── Penalty Calculation ───
        # Punctuation violations: 0.2 each, max 1.0 from punct alone
        punct_penalty = min(1.0, len(violations) * 0.2)
        
        # Grammar issues: 0.15 each, max 0.5 from grammar alone
        grammar_penalty = min(0.5, len(grammar_issues) * 0.15)
        
        # Total technical penalty capped at 1.5 (out of 3.0 technical marks)
        # This leaves at least 1.5/3.0 even for essays with many issues
        total_penalty = min(1.5, punct_penalty + grammar_penalty)
        
        return {
            "penalty": round(total_penalty, 2),
            "violations": violations,
            "grammar_issues": grammar_issues
        }

    def _join_continuation_lines(self, text: str) -> str:
        """
        Join continuation lines back into complete sentences.
        
        OCR text and user input often has line breaks mid-sentence:
        e.g., "ලෙඩ රෝග\nබෝ වෙනවා." should be treated as ONE sentence.
        
        Logic: If a line does NOT end with sentence-ending punctuation,
        join it with the next line.
        """
        lines = text.split('\n')
        joined = []
        buffer = ""
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if buffer:
                    joined.append(buffer)
                    buffer = ""
                continue
            
            if buffer:
                buffer += " " + stripped
            else:
                buffer = stripped
            
            # If this line ends with sentence-ending punctuation, flush the buffer
            if stripped and self.sentence_ender.search(stripped[-1]):
                joined.append(buffer)
                buffer = ""
        
        # Don't forget the last buffer
        if buffer:
            joined.append(buffer)
        
        return '\n'.join(joined)

    def _check_missing_fullstop(self, text: str) -> int:
        """
        Check for sentences that don't end with proper punctuation.
        Splits by newlines and checks each logical sentence.
        
        Returns: count of issues found
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        missing_count = 0
        for line in lines:
            # Only check lines long enough to be a sentence (>15 chars)
            if len(line) > 15:
                stripped = line.rstrip()
                if stripped and not self.sentence_ender.search(stripped[-1]):
                    missing_count += 1
        
        return missing_count

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: HEURISTIC GRAMMAR CHECKS
    # Sinhala grammar without external tools (replaces Sinspell)
    # ═══════════════════════════════════════════════════════════════
    
    def _analyze_grammar(self, text: str) -> List[str]:
        """
        Heuristic grammar analysis for Sinhala essays.
        
        Checks:
        1. Word separation issues (concatenated words → unusually long tokens)
        2. Sentence length anomalies (run-on sentences > 50 words)
        3. Verb marker presence (Sinhala sentences should end with verb markers)
        4. Repeated words (stuttering / copy errors)
        """
        issues = []
        
        # Join continuation lines first for accurate analysis
        joined_text = self._join_continuation_lines(text)
        words = joined_text.split()
        
        if not words:
            return issues
        
        # ─── Check 1: Word Separation Issues ───
        # Sinhala words rarely exceed 25 characters. Long tokens = likely concatenated.
        concat_words = [w for w in words if len(w) > self.max_word_length]
        if concat_words:
            count = len(concat_words)
            issues.append(
                f"Grammar: Word separation issue — {count} word(s) over {self.max_word_length} chars "
                f"(likely concatenated)"
            )
        
        # ─── Check 2: Run-on Sentence Detection ───
        # Split joined text into sentences using sentence enders
        sentences = re.split(r'[.!?।]+', joined_text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            # Only flag truly long sentences (> 50 words without connectors)
            runon_count = 0
            for sent in sentences:
                sent_words = sent.split()
                if len(sent_words) > 50:
                    has_connector = bool(self.sinhala_connectors.search(sent))
                    if not has_connector:
                        runon_count += 1
            
            if runon_count > 0:
                issues.append(
                    f"Grammar: Possible run-on sentence(s) — {runon_count} sentence(s) "
                    f"exceeding 50 words without connectors"
                )
        
        # ─── Check 3: Verb Marker Presence ───
        # In Sinhala, most sentences should end with a verb marker.
        if sentences:
            sentences_with_verbs = 0
            total_checkable = 0
            
            for sent in sentences:
                sent_words = sent.split()
                if len(sent_words) >= 4:  # Only check meaningful sentences (4+ words)
                    total_checkable += 1
                    last_word = sent_words[-1].rstrip('.,!?;:')
                    if self.sinhala_verb_markers.search(last_word):
                        sentences_with_verbs += 1
            
            if total_checkable > 0:
                verb_ratio = sentences_with_verbs / total_checkable
                if verb_ratio < 0.25:  # Very low threshold — only flag truly problematic
                    issues.append(
                        f"Grammar: Low verb marker usage — only {int(verb_ratio * 100)}% of sentences "
                        f"end with recognized verb forms"
                    )
        
        # ─── Check 4: Repeated Words (immediate repetition) ───
        repeat_count = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1] and len(words[i]) > 2:
                repeat_count += 1
        
        if repeat_count > 2:
            issues.append(
                f"Grammar: Repeated words — {repeat_count} immediate word repetition(s)"
            )
        
        # ─── Check 5: Sentence Fragment Detection ───
        if sentences:
            fragment_count = 0
            for sent in sentences:
                if self.sinhala_fragment_enders.search(sent):
                    fragment_count += 1
            
            if fragment_count > 0:
                issues.append(
                    f"Grammar: Incomplete sentence(s) (Fragments) — {fragment_count} sentence(s) "
                    f"end with continuing markers (e.g., 'දී', 'මින්', 'ට') followed by a full stop."
                )
        
        # NOTE: Cohesion/connector analysis removed.
        # Young students (Grade 3-5) write simple sentence-by-sentence essays
        # without connectors. Penalizing for this is unfair.
        
        return issues

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: TOPIC / THEME RELEVANCE
    # Uses DUAL approach: keyword frequency + XLM-R cosine similarity
    # Weighted combination (keyword=60%, cosine=40%)
    # Affects: richness_5
    # ═══════════════════════════════════════════════════════════════
    
    def _compute_keyword_relevance(self, essay_text: str, topic_text: str) -> float:
        """
        Compute keyword-based relevance score.
        
        If the topic word(s) appear frequently in the essay, it's clearly on-topic
        regardless of what the cosine similarity says.
        
        Returns:
            Score from 0.0 to 1.0
        """
        if not topic_text or not essay_text:
            return 0.0
        
        topic_words = topic_text.strip().split()
        total_words = len(essay_text.split())
        
        if total_words == 0:
            return 0.0
        
        # ─── Filter Stopwords ───
        # We want to find the CORE nouns (e.g., "තාත්තා", "ගස", "පවුල")
        # not filler words like "මගේ" (My) or "ගැන" (About).
        significant_keywords = [
            tw for tw in topic_words 
            if tw.strip() not in self.sinhala_topic_stopwords
        ]
        
        # If the topic ONLY consists of stopwords (unlikely but possible),
        # fall back to the full list. Otherwise, use the significant ones.
        search_terms = significant_keywords if significant_keywords else topic_words
        print(f"[HYBRID] Significant topic words: {search_terms}")

        # Count occurrences of search terms (and common variants)
        total_hits = 0
        for tw in search_terms:
            tw_clean = tw.strip()
            if not tw_clean:
                continue
            
            # Count exact matches and substring matches
            # e.g., topic "පිරිසිදුකම" should match "පිරිසිඳු", "අපිරිසිඳු" etc.
            # Use the root form (first 4+ chars for Sinhala)
            root = tw_clean[:min(6, len(tw_clean))] if len(tw_clean) > 3 else tw_clean
            
            # Count how many words in the essay contain this root
            hits = sum(1 for w in essay_text.split() if root in w)
            total_hits += hits
        
        # Compute relevance based on frequency
        # 5+ mentions in 150 words = clearly on-topic
        if not significant_keywords and total_hits > 0:
            # If we only had stopwords, penalize the certainty
            return 0.5
            
        if total_hits >= 8:
            return 1.0   # Extremely on-topic
        elif total_hits >= 5:
            return 0.9   # Very on-topic
        elif total_hits >= 3:
            return 0.7   # Clearly on-topic
        elif total_hits >= 1:
            return 0.5   # Topic mentioned
        else:
            return 0.0   # Topic not mentioned at all
    
    def compute_theme_relevance(
        self,
        essay_cls: torch.Tensor,
        topic_text: str,
        model,
        tokenizer,
        device,
        essay_text: str = ""
    ) -> float:
        """
        Calculate theme relevance using DUAL weighted approach:
        1. Keyword frequency (fast, exact)  — 60% weight
        2. Cosine similarity (semantic)     — 40% weight
        
        WHY WEIGHTED instead of MAX:
            XLM-R CLS embeddings produce cosine similarity of ~0.65-0.80
            for ANY Sinhala text pair regardless of topic. Using MAX would
            let this inflated cosine override a correct keyword=0 signal,
            causing off-topic essays to escape penalty.
        
        Examples:
            On-topic  "පිරිසිදුකම": keyword=1.0 → 1.0×0.6 + 0.76×0.4 = 0.90 → no penalty
            Off-topic "ක්‍රීඩා":     keyword=0.0 → 0.0×0.6 + 0.68×0.4 = 0.27 → penalty
        
        Args:
            essay_cls: CLS embedding from the essay encoding
            topic_text: The specified topic / prompt
            model: The XLM-R model with encoder access
            tokenizer: The tokenizer
            device: torch device
            essay_text: Raw essay text for keyword matching
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not topic_text or not topic_text.strip():
            return 1.0  # Perfect relevance if no topic specified

        # ─── Signal 1: Keyword Frequency ───
        keyword_score = self._compute_keyword_relevance(essay_text, topic_text)
        print(f"[HYBRID] Keyword Relevance: {keyword_score:.2f}")

        # ─── Signal 2: Cosine Similarity ───
        # Expand short topics for better CLS embedding
        topic_words = topic_text.strip().split()
        if len(topic_words) <= 2:
            expanded_topic = f"{topic_text} ගැන රචනය. {topic_text} පිළිබඳ."
            print(f"[HYBRID] Short topic expanded: '{topic_text}' → '{expanded_topic}'")
        else:
            expanded_topic = topic_text

        t_enc = tokenizer(
            expanded_topic,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            t_out = model.encoder(**t_enc)
            topic_cls = t_out.last_hidden_state[:, 0, :]

        sim = torch.nn.functional.cosine_similarity(essay_cls, topic_cls).item()
        cosine_score = max(0.0, sim)
        print(f"[HYBRID] Cosine Similarity: {cosine_score:.4f}")

        # ─── Combined Score: Weighted (keyword=60%, cosine=40%) ───
        # Keyword is the primary signal (reliable, exact match).
        # Cosine is secondary (helps with synonyms/paraphrasing but
        # is inflated for same-language text in XLM-R).
        KEYWORD_WEIGHT = 0.6
        COSINE_WEIGHT = 0.4
        
        final_score = (keyword_score * KEYWORD_WEIGHT) + (cosine_score * COSINE_WEIGHT)
        print(f"[HYBRID] Final Relevance (keyword×{KEYWORD_WEIGHT} + cosine×{COSINE_WEIGHT}): {final_score:.3f}")
        
        return final_score

    def compute_richness_penalty(
        self,
        relevance_score: float,
        word_count: int
    ) -> Dict[str, float]:
        """
        Calculate combined richness penalty from theme relevance and word count.
        
        Marking Scheme Requirements:
        - Topic relevance is critical for richness_5
        - Minimum 150 words required
        
        Returns:
            {
                "theme_penalty": float (0..2.0),
                "word_count_penalty": float (0..1.5),
                "total_richness_penalty": float (0..3.5)
            }
        """
        # ─── Theme Relevance Penalty ───
        # Calibrated for dual keyword+cosine approach
        # 0.55+ means either keyword frequency OR cosine similarity is strong
        if relevance_score >= 0.55:
            theme_penalty = 0.0   # Clearly relevant → no penalty
        elif relevance_score >= 0.35:
            theme_penalty = 0.5   # Somewhat relevant → mild penalty
        elif relevance_score >= 0.2:
            theme_penalty = 1.0   # Partially off-topic → moderate penalty
        else:
            theme_penalty = 2.0   # Off-topic → severe penalty
        
        # ─── Word Count Penalty (Marking Scheme: min 150 words) ───
        if word_count >= 150:
            wc_penalty = 0.0
        elif word_count >= 100:
            # Between 100-149 words: proportional penalty
            wc_penalty = (150 - word_count) / 150 * 1.0  # Up to ~0.33
        elif word_count >= 50:
            # Between 50-99 words: significant penalty
            wc_penalty = 0.5 + (100 - word_count) / 100 * 0.5  # 0.5 to 1.0
        else:
            # Below 50 words: severe penalty
            wc_penalty = 1.5  # Maximum word count penalty
        
        return {
            "theme_penalty": round(theme_penalty, 2),
            "word_count_penalty": round(wc_penalty, 2),
            "total_richness_penalty": round(min(3.5, theme_penalty + wc_penalty), 2)
        }


# Module-level singleton
rubric_evaluator = RubricEvaluator()
