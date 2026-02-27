"""
Query Normalization Benchmark
==============================
Benchmarks multiple normalization approaches on the generated dataset.

Normalizers:
  1. Identity              - baseline, no change
  2. PySpellChecker        - token-by-token spell correction (current approach)
  3. SymSpell              - faster, supports compound word correction
  4. Rules                 - regex + entity canonicalization (flight IDs, stock tickers, product spacing)
  5. RapidFuzz             - fuzzy brand name matching
  6. Combined              - Rules → SymSpell → RapidFuzz pipeline
  --- ML ---
  7. ContextualSpellCheck  - spaCy pipeline with BERT contextual embeddings
  8. T5SpellCorrector      - HuggingFace T5 fine-tuned for spelling correction
  9. CombinedML            - Rules → T5 pipeline (entity rules first, T5 for the rest)

Metrics (per normalizer, per category):
  exact_match         - % where output == canonical (case-insensitive)
  cer                 - character error rate: edit_dist / max(len_pred, len_gold)
  wer                 - word error rate: token-level edit distance / n_gold_tokens
  no_change_precision - on no_change rows: % correctly left unchanged
  over_correction     - on no_change rows: % wrongly changed
  latency_mean_ms     - mean per-query latency
  latency_p50_ms      - p50 latency
  latency_p95_ms      - p95 latency
  latency_p99_ms      - p99 latency

Usage:
  pip install -r requirements.txt
  python3 benchmark.py [--dataset dataset.csv]
"""

import re
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional

warnings.filterwarnings("ignore")

# ── Optional imports ───────────────────────────────────────────────────────────

try:
    from Levenshtein import distance as _lev
    def edit_distance(a: str, b: str) -> int: return _lev(a, b)
except ImportError:
    # Pure-python fallback
    def edit_distance(a: str, b: str) -> int:
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, n + 1):
                dp[j] = prev[j - 1] if a[i-1] == b[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
        return dp[n]

try:
    from spellchecker import SpellChecker as _SC
    HAS_PYSPELL = True
except ImportError:
    HAS_PYSPELL = False
    print("Warning: pyspellchecker not installed — skipping PySpell normalizer")

try:
    from symspellpy import SymSpell as _SS, Verbosity as _V
    import pkg_resources
    HAS_SYMSPELL = True
except ImportError:
    HAS_SYMSPELL = False
    print("Warning: symspellpy not installed — skipping SymSpell normalizer")

try:
    from rapidfuzz import process as _rf_process, fuzz as _rf_fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("Warning: rapidfuzz not installed — skipping RapidFuzz normalizer")

try:
    import spacy as _spacy
    import contextualSpellCheck as _csc
    _csc_nlp = _spacy.load("en_core_web_sm")
    _csc.add_to_pipe(_csc_nlp)
    HAS_CONTEXTUAL = True
except Exception:
    HAS_CONTEXTUAL = False
    print("Warning: contextualSpellCheck/spacy not available — skipping ContextualSpellCheck normalizer")
    print("  Install: pip install contextualSpellCheck && python -m spacy download en_core_web_sm")

try:
    from transformers import pipeline as _hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed — skipping T5 normalizer")
    print("  Install: pip install transformers torch")

# ── Brand list for fuzzy matching ──────────────────────────────────────────────

BRANDS = [
    "amazon", "google", "facebook", "twitter", "instagram", "youtube",
    "linkedin", "reddit", "netflix", "spotify", "microsoft", "adobe",
    "dropbox", "github", "slack", "zoom", "paypal", "ebay", "walmart",
    "target", "best buy", "new york times", "bbc", "cnn", "espn",
    "gmail", "outlook", "yahoo", "apple", "samsung", "dell", "hp",
    "lenovo", "asus", "acer", "toshiba", "sony", "lg", "panasonic",
    "booking.com", "expedia", "airbnb", "tripadvisor", "yelp",
    "doordash", "ubereats", "grubhub", "lyft", "uber",
    "twitch", "discord", "telegram", "whatsapp", "snapchat", "tiktok",
]

# ── Entity lists for rules normalizer ──────────────────────────────────────────

# Common IATA codes (2-3 letter airline codes)
IATA_CODES = {
    "AA", "BA", "DL", "UA", "LH", "AF", "EK", "QR", "SQ", "CX",
    "VS", "KL", "IB", "TK", "AC", "QF", "NH", "JL", "MH", "TG",
    "AI", "SA", "ET", "KE", "OZ", "CI", "BR", "LA", "AV", "AM",
    "WN", "B6", "AS", "F9", "NK", "G4", "VX", "HA",
}

# Common stock tickers → company name aliases
STOCK_ALIASES: dict[str, list[str]] = {
    "AAPL": ["apple", "aapl"],
    "TSLA": ["tesla", "tsla"],
    "MSFT": ["microsoft", "msft"],
    "GOOGL": ["google", "alphabet", "googl"],
    "AMZN": ["amazon", "amzn"],
    "META": ["meta", "facebook", "fb"],
    "NVDA": ["nvidia", "nvda"],
    "NFLX": ["netflix", "nflx"],
    "PYPL": ["paypal", "pypl"],
    "SNAP": ["snapchat", "snap"],
    "AMD":  ["amd"],
    "INTC": ["intel", "intc"],
    "UBER": ["uber"],
    "LYFT": ["lyft"],
    "ABNB": ["airbnb", "abnb"],
    "COIN": ["coinbase", "coin"],
    "HOOD": ["robinhood", "hood"],
}

# Reverse map: alias → ticker
_ALIAS_TO_TICKER: dict[str, str] = {}
for ticker, aliases in STOCK_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_TICKER[alias.lower()] = ticker

# Product model patterns: brand → canonical prefix
PRODUCT_BRANDS = ["iphone", "samsung", "macbook", "ipad", "pixel", "surface"]

# ── Base normalizer ────────────────────────────────────────────────────────────

class Normalizer(ABC):
    name: str

    def warmup(self) -> None:
        """Called once before benchmarking to initialize any lazy state."""
        pass

    @abstractmethod
    def normalize(self, query: str) -> str:
        ...

    def normalize_batch(self, queries: list[str]) -> list[str]:
        return [self.normalize(q) for q in queries]


# ── 1. Identity (baseline) ────────────────────────────────────────────────────

class IdentityNormalizer(Normalizer):
    name = "Identity (baseline)"

    def normalize(self, query: str) -> str:
        return query


# ── 2. PySpellChecker ────────────────────────────────────────────────────────

class PySpellNormalizer(Normalizer):
    name = "PySpellChecker"

    def __init__(self):
        if not HAS_PYSPELL:
            raise RuntimeError("pyspellchecker not installed")
        self._sc = _SC()

    def normalize(self, query: str) -> str:
        words = query.lower().split()
        return " ".join(self._sc.correction(w) or w for w in words)


# ── 3. SymSpell ───────────────────────────────────────────────────────────────

_ORCAS_VOCAB = Path(__file__).parent / "orcas_vocab.txt"


class SymSpellNormalizer(Normalizer):
    name = "SymSpell"

    def __init__(self, max_edit_distance: int = 2):
        if not HAS_SYMSPELL:
            raise RuntimeError("symspellpy not installed")
        self._sym = _SS(max_dictionary_edit_distance=max_edit_distance)
        # Try importlib.resources first (works in newer Python/packaging setups),
        # fall back to pkg_resources for older environments.
        _dict_loaded = False
        # Try candidate dictionary filenames (name changed across symspellpy versions)
        _DICT_CANDIDATES = ["frequency_dictionary_en_82_765.txt", "en-80k.txt"]
        try:
            import importlib.resources as _ir
            for _fname in _DICT_CANDIDATES:
                try:
                    _ref = _ir.files("symspellpy").joinpath(_fname)
                    with _ir.as_file(_ref) as _dp:
                        _dict_loaded = self._sym.load_dictionary(str(_dp), term_index=0, count_index=1)
                    if _dict_loaded:
                        break
                except Exception:
                    pass
        except Exception:
            pass
        if not _dict_loaded:
            for _fname in _DICT_CANDIDATES:
                _dp = pkg_resources.resource_filename("symspellpy", _fname)
                _dict_loaded = self._sym.load_dictionary(_dp, term_index=0, count_index=1)
                if _dict_loaded:
                    break
        if _ORCAS_VOCAB.exists():
            self._sym.load_dictionary(str(_ORCAS_VOCAB), term_index=0, count_index=1)
            self.name = "SymSpell+ORCAS"
        self._max_ed = max_edit_distance

    def normalize(self, query: str) -> str:
        # Use lookup_compound for multi-token correction
        suggestions = self._sym.lookup_compound(
            query.lower(), max_edit_distance=self._max_ed
        )
        if suggestions:
            return suggestions[0].term
        return query.lower()


# ── 4. Rules (entity + regex) ────────────────────────────────────────────────

class RulesNormalizer(Normalizer):
    name = "Rules (entity + regex)"

    # Flight: digits + IATA  or  IATA + digits  →  IATA + digits (no space)
    _FLIGHT_LOOSE = re.compile(
        r'\b(?:flight\s+)?(\d{2,4})\s*([A-Z]{2,3})\b'   # 163 SQ
        r'|'
        r'\b(?:flight\s+)?([A-Z]{2,3})\s+(\d{2,4})\b',  # SQ 163  (space)
        re.IGNORECASE
    )

    # Product spacing: brand directly followed by digits/variant ("iphone15")
    _PRODUCT_SPACING = re.compile(
        r'\b(iphone|macbook|ipad|pixel|galaxy|surface|airpods)'
        r'(\d+|pro|air|mini|max|ultra|plus)\b',
        re.IGNORECASE
    )

    # Stock: remove surrounding noise, keep just the ticker
    _STOCK_NOISE = re.compile(
        r'\b(stock|share|price|shares|equity|ticker|market|trading|invest(?:ment)?)\b',
        re.IGNORECASE
    )

    # Common compound words that users type without a space.
    # Applied per-token so works in multi-token queries too
    # e.g. "restarants nearme" → "restarants near me" (then GuardedPySpell fixes "restarants")
    _COMPOUND_SPLITS: dict[str, str] = {
        "nearme":       "near me",
        "nearbyme":     "near by me",
        "newyork":      "new york",
        "losangeles":   "los angeles",
        "sanfrancisco": "san francisco",
        "lasvegас":     "las vegas",
        "bestbuy":      "best buy",
        "homedepot":    "home depot",
        "wholefoods":   "whole foods",
        "starbucks":    "starbucks",   # already one word, no-op
        "doordash":     "doordash",
        "ubereats":     "uber eats",
        "grubhub":      "grubhub",
        "openai":       "openai",
        "chatgpt":      "chatgpt",
        "youtube":      "youtube",
        "facebook":     "facebook",
        "instagram":    "instagram",
        "whatsapp":     "whatsapp",
        "linkedin":     "linkedin",
        "tiktok":       "tiktok",
    }

    def _normalize_flight(self, query: str) -> str:
        q_upper = query.upper()
        def _repl(m):
            if m.group(1):   # digits IATA
                num, code = m.group(1), m.group(2).upper()
            else:            # IATA digits
                code, num = m.group(3).upper(), m.group(4)
            if code in IATA_CODES:
                return f"{code}{num}"
            return m.group(0)
        result = self._FLIGHT_LOOSE.sub(_repl, query)
        return result

    def _normalize_stock(self, query: str) -> Optional[str]:
        ql = query.lower().strip()
        tokens = ql.split()
        # Check if any token is a known ticker or alias
        found_ticker = None
        for tok in tokens:
            # Direct ticker match (uppercase)
            if tok.upper() in STOCK_ALIASES:
                found_ticker = tok.upper()
                break
            # Alias match
            if tok in _ALIAS_TO_TICKER:
                found_ticker = _ALIAS_TO_TICKER[tok]
        if found_ticker:
            # Case 1: stock noise words present (e.g. "AAPL stock price")
            remaining = self._STOCK_NOISE.sub("", ql).strip()
            if remaining != ql.strip():
                return found_ticker
            # Case 2: explicit ticker token present alongside alias
            # (e.g. "apple aapl", "google GOOGL") — but NOT "google pixel 8"
            if found_ticker.lower() in tokens:
                return found_ticker
        return None

    def _normalize_product_spacing(self, query: str) -> str:
        return self._PRODUCT_SPACING.sub(lambda m: f"{m.group(1)} {m.group(2)}", query)

    def _normalize_compounds(self, query: str) -> str:
        """Split known compound tokens anywhere in the query.
        Works per-token so handles mixed queries like 'restarants nearme'."""
        tokens = query.lower().split()
        return " ".join(self._COMPOUND_SPLITS.get(tok, tok) for tok in tokens)

    def _normalize_word_order(self, query: str) -> str:
        """Reorder product queries so the brand/product-line token comes first.

        Handles patterns like:
          's24 samsung'      → 'samsung s24'
          'pro 14 macbook'   → 'macbook pro 14'
          'ultra s23 samsung'→ 'samsung ultra s23'
        """
        tokens = query.lower().split()
        if len(tokens) < 2:
            return query
        # Find a PRODUCT_BRANDS token that is not already at position 0
        for i, tok in enumerate(tokens):
            if i > 0 and tok in PRODUCT_BRANDS:
                # Move brand to front, preserve relative order of the rest
                return " ".join([tok] + tokens[:i] + tokens[i + 1:])
        return query

    def normalize(self, query: str) -> str:
        q = query.strip()

        # 1. Stock canonicalization
        stock = self._normalize_stock(q)
        if stock:
            return stock

        # 2. Flight ID normalization
        q = self._normalize_flight(q)

        # 3. Compound splitting (nearme → near me, newyork → new york)
        q = self._normalize_compounds(q)

        # 4. Product spacing
        q = self._normalize_product_spacing(q)

        # 5. Product word order
        q = self._normalize_word_order(q)

        # 6. Clean up extra whitespace
        q = re.sub(r'\s+', ' ', q).strip()

        return q


# ── 5. RapidFuzz (brand matching) ────────────────────────────────────────────

class RapidFuzzNormalizer(Normalizer):
    name = "RapidFuzz (brand match)"

    def __init__(self, score_cutoff: int = 82):
        if not HAS_RAPIDFUZZ:
            raise RuntimeError("rapidfuzz not installed")
        self._cutoff = score_cutoff

    def normalize(self, query: str) -> str:
        ql = query.lower().strip()

        # Only attempt brand correction on short queries (≤ 3 tokens)
        tokens = ql.split()
        if len(tokens) > 3:
            return query

        # Skip very short queries — too ambiguous to fuzzy-match safely
        # (e.g. 'appl', 'npm', 'gcc' should not be matched to brand names)
        if len(ql) <= 5:
            return query

        # Try matching each n-gram of the query against the brand list
        # First try the full query, then try progressively smaller windows
        result = _rf_process.extractOne(
            ql, BRANDS,
            scorer=_rf_fuzz.token_sort_ratio,
            score_cutoff=self._cutoff,
        )
        if result:
            best_match, score, _ = result
            return best_match

        return query


# ── 6. Combined ───────────────────────────────────────────────────────────────

class CombinedNormalizer(Normalizer):
    name = "Combined (Rules + SymSpell + RapidFuzz)"

    def __init__(self):
        self._rules = RulesNormalizer()
        self._symspell = SymSpellNormalizer() if HAS_SYMSPELL else None
        self._rfuzz   = RapidFuzzNormalizer() if HAS_RAPIDFUZZ else None

    def normalize(self, query: str) -> str:
        q = query.strip()

        # Step 1: Apply entity/structural rules first (highest precision)
        q_rules = self._rules.normalize(q)
        if q_rules.lower() != q.lower():
            return q_rules  # Rules made a change — trust it

        # Step 2: SymSpell for general typo correction
        if self._symspell:
            q_sym = self._symspell.normalize(q)
            if q_sym.lower() != q.lower():
                return q_sym

        # Step 3: RapidFuzz for brand name typos (catches what SymSpell misses
        # on compound brand names like "bestbuyt" → "best buy")
        if self._rfuzz:
            q_rf = self._rfuzz.normalize(q)
            if q_rf.lower() != q.lower():
                return q_rf

        return q


# ── 7. GuardedPySpell ────────────────────────────────────────────────────────

class GuardedPySpellNormalizer(Normalizer):
    """PySpellChecker with guards to prevent over-correction.

    PySpellChecker gets 88% on single_typo and 71% on multi_typo, but has
    40% over-correction on no-change queries (e.g. 'appl' → 'apple').

    Guards:
      - Skip tokens ≤ 4 chars  (appl, npm, gcc, css, java, rust, echo, go)
      - Skip all-uppercase tokens  (AAPL, NYC, SQ — abbreviations/tickers)

    Most legitimate short abbreviations are ≤ 4 chars or all-caps.
    Typos worth correcting are almost always ≥ 5 chars ('wheather', 'suhsi').
    """
    name = "PySpell (guarded)"

    def __init__(self):
        if not HAS_PYSPELL:
            raise RuntimeError("pyspellchecker not installed")
        self._sc = _SC()

    def _skip(self, token: str) -> bool:
        return len(token) <= 4 or token.isupper()

    def normalize(self, query: str) -> str:
        words = query.lower().split()
        return " ".join(
            w if self._skip(w) else (self._sc.correction(w) or w)
            for w in words
        )


# ── 8. CombinedV2 (Rules + GuardedPySpell + RapidFuzz) ───────────────────────

class CombinedV2Normalizer(Normalizer):
    """Improved pipeline: Rules → RapidFuzz (single-token) → SymSpell split → GuardedPySpell → RapidFuzz (multi-token).

    Rules handles structured entities (flight IDs, stock tickers, product
    spacing/order) with perfect precision. RapidFuzz runs first on single-token
    queries to catch brand typos (bestbuyt→best buy) before SymSpell can corrupt
    them (bestbuyt→best but). SymSpell compound splitting then handles concatenated
    words (nearme→near me). GuardedPySpell handles general typos while protecting
    short tokens. RapidFuzz runs again at the end for multi-token brand typos.
    """
    name = "CombinedV2 (Rules + GuardedPySpell + RapidFuzz)"

    def __init__(self):
        self._rules    = RulesNormalizer()
        self._symspell = SymSpellNormalizer() if HAS_SYMSPELL else None
        self._pyspell  = GuardedPySpellNormalizer() if HAS_PYSPELL else None
        self._rfuzz    = RapidFuzzNormalizer() if HAS_RAPIDFUZZ else None

    def normalize(self, query: str) -> str:
        q = query.strip()

        # Step 1: Rules — flight IDs, stock tickers, product spacing/order
        q_rules = self._rules.normalize(q)
        if q_rules.lower() != q.lower():
            return q_rules

        # Step 2: RapidFuzz — brand name typos for single-token queries.
        # Must run before SymSpell compound splitting: SymSpell splits 'bestbuyt'
        # into 'best but' (wrong) whereas RapidFuzz correctly maps it to 'best buy'.
        if self._rfuzz and ' ' not in q:
            q_rf = self._rfuzz.normalize(q)
            if q_rf.lower() != q.lower():
                return q_rf

        # Step 3: SymSpell compound splitting for single-token queries only.
        # Only accept if SymSpell introduces a space (compound split).
        # Known compounds (nearme, newyork etc.) are handled by Rules above,
        # so this catches any remaining edge cases for single-token inputs.
        if self._symspell and ' ' not in q:
            q_sym = self._symspell.normalize(q)
            if ' ' in q_sym:
                return q_sym

        # Step 4: GuardedPySpell — general typos (skips short/uppercase tokens)
        if self._pyspell:
            q_spell = self._pyspell.normalize(q)
            if q_spell.lower() != q.lower():
                return q_spell

        # Step 5: RapidFuzz — brand name typos for multi-token queries
        # (e.g. 'gooogle maps' → 'google maps', 'spotifiy premium' → 'spotify premium')
        if self._rfuzz:
            q_rf = self._rfuzz.normalize(q)
            if q_rf.lower() != q.lower():
                return q_rf

        return q


# ── 9. ContextualSpellCheck (spaCy + BERT) ───────────────────────────────────

class ContextualSpellCheckNormalizer(Normalizer):
    """Uses BERT contextual embeddings to decide whether and how to correct
    each token. Unlike SymSpell, it sees the full query context before
    making a correction — so 'appl' in an ambiguous context stays as-is,
    while 'wheather nyc' correctly becomes 'weather nyc'.

    Requires:
      pip install contextualSpellCheck
      python -m spacy download en_core_web_sm
    """
    name = "ContextualSpellCheck (BERT)"

    def __init__(self):
        if not HAS_CONTEXTUAL:
            raise RuntimeError("contextualSpellCheck not available")
        self._nlp = _csc_nlp

    def normalize(self, query: str) -> str:
        doc = self._nlp(query)
        # doc._.outcome_spellCheck is the full corrected string
        result = doc._.outcome_spellCheck
        return result if result else query


# ── 8. T5 Spell Corrector (HuggingFace) ──────────────────────────────────────

class T5SpellCorrector(Normalizer):
    """Fine-tuned T5 model for spelling correction.
    Model: oliverguhr/spelling-correction-english-base

    This is a seq2seq model trained on noisy→clean sentence pairs.
    It handles multi-token typos, word order, and spacing better than
    dictionary-based approaches, but at significantly higher latency.

    Expected latency: ~100–500ms on CPU, ~20–80ms on GPU.

    Requires:
      pip install transformers torch (or transformers sentencepiece)
    """
    name = "T5 (oliverguhr/spelling-correction)"

    _MODEL_ID = "oliverguhr/spelling-correction-english-base"

    def __init__(self):
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers not installed")
        self._pipe = None  # lazy load in warmup()

    def warmup(self) -> None:
        print(f"    Loading {self._MODEL_ID}...", end=" ", flush=True)
        self._pipe = _hf_pipeline(
            "text2text-generation",
            model=self._MODEL_ID,
            tokenizer=self._MODEL_ID,
        )
        # Prime the model with a dummy query
        self._pipe("warmup query", max_length=64)
        print("ready")

    def normalize(self, query: str) -> str:
        if self._pipe is None:
            self.warmup()
        result = self._pipe(query, max_length=128, num_beams=4)
        return result[0]["generated_text"].strip()


# ── 9. CombinedML (Rules → T5) ───────────────────────────────────────────────

class CombinedMLNormalizer(Normalizer):
    """Best-of-both-worlds pipeline:
      1. Rules handle structured entity normalization (flight IDs, stock tickers,
         product model reordering) with zero latency and perfect precision.
      2. T5 handles everything else — general typos, multi-token corrections,
         brand names — using full-query context.

    This avoids running T5 on queries that rules already handle perfectly,
    saving latency on the most common structured patterns.
    """
    name = "CombinedML (Rules → T5)"

    def __init__(self):
        self._rules = RulesNormalizer()
        self._t5    = T5SpellCorrector() if HAS_TRANSFORMERS else None

    def warmup(self) -> None:
        if self._t5:
            self._t5.warmup()

    def normalize(self, query: str) -> str:
        # Step 1: Rules first — highest precision for structured entities
        q_rules = self._rules.normalize(query)
        if q_rules.lower() != query.lower():
            return q_rules

        # Step 2: T5 for everything else
        if self._t5:
            return self._t5.normalize(query)

        return query


# ── Metrics ───────────────────────────────────────────────────────────────────

def char_error_rate(pred: str, gold: str) -> float:
    """CER = edit_distance / max(len(pred), len(gold))."""
    if not pred and not gold:
        return 0.0
    return edit_distance(pred.lower(), gold.lower()) / max(len(pred), len(gold))


def word_error_rate(pred: str, gold: str) -> float:
    """WER = token-level edit distance / number of gold tokens."""
    pred_toks = pred.lower().split()
    gold_toks = gold.lower().split()
    if not gold_toks:
        return 0.0
    m, n = len(pred_toks), len(gold_toks)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if pred_toks[i-1] == gold_toks[j-1] \
                    else 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n] / n


def run_benchmark(normalizer: Normalizer, df: pd.DataFrame, n_timing_reps: int = 5) -> dict:
    """Run a normalizer on the dataset and return metrics."""
    queries = df["noisy"].tolist()

    # ── Timing ───────────────────────────────────────────────────────────────
    latencies_ms = []
    for q in queries:
        t0 = time.perf_counter()
        for _ in range(n_timing_reps):
            normalizer.normalize(q)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) / n_timing_reps * 1000)

    # ── Predictions ──────────────────────────────────────────────────────────
    preds = [normalizer.normalize(q) for q in queries]
    df = df.copy()
    df["pred"] = preds

    def em(row): return row["pred"].lower().strip() == row["canonical"].lower().strip()
    def cer(row): return char_error_rate(row["pred"], row["canonical"])
    def wer(row): return word_error_rate(row["pred"], row["canonical"])

    df["em"]  = df.apply(em, axis=1)
    df["cer"] = df.apply(cer, axis=1)
    df["wer"] = df.apply(wer, axis=1)

    # No-change precision and over-correction rate
    nc = df[~df["should_change"]]
    no_change_precision = (nc["pred"].str.lower().str.strip() == nc["noisy"].str.lower().str.strip()).mean() if len(nc) else float("nan")
    over_correction     = 1.0 - no_change_precision if not np.isnan(no_change_precision) else float("nan")

    # ── Per-category exact match ──────────────────────────────────────────────
    cat_em = df.groupby("category")["em"].mean().to_dict()

    return {
        "name":                normalizer.name,
        "exact_match":         df["em"].mean(),
        "cer_mean":            df["cer"].mean(),
        "wer_mean":            df["wer"].mean(),
        "no_change_precision": no_change_precision,
        "over_correction":     over_correction,
        "latency_mean_ms":     np.mean(latencies_ms),
        "latency_p50_ms":      np.percentile(latencies_ms, 50),
        "latency_p95_ms":      np.percentile(latencies_ms, 95),
        "latency_p99_ms":      np.percentile(latencies_ms, 99),
        "per_category":        cat_em,
        "_df":                 df,      # store for detailed output
        "_latencies":          latencies_ms,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(Path(__file__).parent / "dataset.csv"))
    parser.add_argument("--reps",    type=int, default=5, help="Timing repetitions per query")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    print(f"Loaded {len(df)} rows from {args.dataset}")
    print(f"Categories: {df['category'].value_counts().to_dict()}\n")

    # ── Build normalizer list ─────────────────────────────────────────────────
    normalizers: list[Normalizer] = [IdentityNormalizer(), RulesNormalizer()]
    if HAS_PYSPELL:
        normalizers.append(PySpellNormalizer())
    if HAS_SYMSPELL:
        normalizers.append(SymSpellNormalizer())
    if HAS_RAPIDFUZZ:
        normalizers.append(RapidFuzzNormalizer())
    if HAS_SYMSPELL and HAS_RAPIDFUZZ:
        normalizers.append(CombinedNormalizer())
    if HAS_PYSPELL:
        normalizers.append(GuardedPySpellNormalizer())
    if HAS_PYSPELL and HAS_RAPIDFUZZ:
        normalizers.append(CombinedV2Normalizer())
    # ML normalizers (disabled — too slow and underperform rules-based)
    # if HAS_CONTEXTUAL:
    #     normalizers.append(ContextualSpellCheckNormalizer())
    # if HAS_TRANSFORMERS:
    #     normalizers.append(T5SpellCorrector())
    #     normalizers.append(CombinedMLNormalizer())

    # Warmup
    for norm in normalizers:
        norm.warmup()

    # ── Run benchmarks ────────────────────────────────────────────────────────
    results = []
    for norm in normalizers:
        print(f"Benchmarking: {norm.name}...", end=" ", flush=True)
        r = run_benchmark(norm, df, n_timing_reps=args.reps)
        results.append(r)
        print(f"EM={r['exact_match']:.1%}  CER={r['cer_mean']:.3f}  lat_p50={r['latency_p50_ms']:.2f}ms")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*90)
    print("SUMMARY — Overall Metrics")
    print("="*90)

    summary_rows = []
    for r in results:
        summary_rows.append({
            "Normalizer":         r["name"],
            "Exact Match":        f"{r['exact_match']:.1%}",
            "CER":                f"{r['cer_mean']:.3f}",
            "WER":                f"{r['wer_mean']:.3f}",
            "No-change Prec.":    f"{r['no_change_precision']:.1%}" if not np.isnan(r['no_change_precision']) else "N/A",
            "Over-correction":    f"{r['over_correction']:.1%}"     if not np.isnan(r['over_correction'])     else "N/A",
            "Lat mean (ms)":      f"{r['latency_mean_ms']:.2f}",
            "Lat p50 (ms)":       f"{r['latency_p50_ms']:.2f}",
            "Lat p95 (ms)":       f"{r['latency_p95_ms']:.2f}",
            "Lat p99 (ms)":       f"{r['latency_p99_ms']:.2f}",
        })

    try:
        from tabulate import tabulate
        print(tabulate(summary_rows, headers="keys", tablefmt="rounded_outline"))
    except ImportError:
        pd.DataFrame(summary_rows).to_string(index=False)
        print(pd.DataFrame(summary_rows).to_string(index=False))

    # ── Per-category table ────────────────────────────────────────────────────
    categories = sorted(df["category"].unique())
    print("\n" + "="*90)
    print("PER-CATEGORY Exact Match")
    print("="*90)

    cat_rows = []
    for r in results:
        row = {"Normalizer": r["name"][:30]}
        for cat in categories:
            row[cat] = f"{r['per_category'].get(cat, float('nan')):.0%}"
        cat_rows.append(row)

    try:
        from tabulate import tabulate
        print(tabulate(cat_rows, headers="keys", tablefmt="rounded_outline"))
    except ImportError:
        print(pd.DataFrame(cat_rows).to_string(index=False))

    # ── Sample predictions ────────────────────────────────────────────────────
    print("\n" + "="*90)
    print("SAMPLE PREDICTIONS — Combined vs Identity (first 5 per category)")
    print("="*90)

    combined_r = next((r for r in results if "CombinedV2" in r["name"]),
                      next((r for r in results if "Combined" in r["name"]), results[-1]))
    identity_r = results[0]

    for cat in categories:
        sub   = combined_r["_df"][combined_r["_df"]["category"] == cat].head(5)
        id_sub = identity_r["_df"][identity_r["_df"]["category"] == cat].head(5)
        print(f"\n  {cat.upper()}")
        print(f"  {'Noisy':<30} {'Canonical':<25} {'Combined pred':<25} {'EM':>4}")
        print(f"  {'-'*30} {'-'*25} {'-'*25} {'-'*4}")
        for (_, row), (_, id_row) in zip(sub.iterrows(), id_sub.iterrows()):
            em_mark = "✓" if row["em"] else "✗"
            print(f"  {row['noisy']:<30} {row['canonical']:<25} {row['pred']:<25} {em_mark:>4}")

    # ── Save full results ─────────────────────────────────────────────────────
    out_path = Path(args.dataset).parent / "results.csv"
    combined_r["_df"].to_csv(out_path, index=False)
    print(f"\nFull predictions saved to {out_path}")


if __name__ == "__main__":
    main()
