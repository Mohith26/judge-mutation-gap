"""Synthetic perturbation families, from naive to context-free swaps.

- negation: flip the claim's polarity with a rule (the 'flip yes/no' family).
- random_swap: replace the clinical answer span with the answer span of a *different*
  patient with the same error type (plausible medicine, incoherent with this chart).
The real family is the clinician-authored MEDEC error, which is a plausible MedQA
distractor chosen for this patient.
"""
from __future__ import annotations

import random
import re

_AUX = re.compile(r"\b(is|are|was|were|has been|have been|will be)\b", re.IGNORECASE)


def negate(claim: str) -> str:
    m = _AUX.search(claim)
    if m:
        return claim[: m.end()] + " not" + claim[m.end():]
    m = re.search(r"\b(prescribed|given|started|advised|diagnosed|treated|suspected|recommended|administered)\b", claim, re.IGNORECASE)
    if m:
        return claim[: m.start()] + "not " + claim[m.start():]
    return "It is not the case that " + claim[0].lower() + claim[1:]


def _tok(s: str) -> list[str]:
    return s.split()


FUNC = {"with", "of", "is", "are", "was", "were", "a", "an", "the", "by", "to", "for", "after",
        "and", "or", "be", "as", "in", "on", "from", "due", "because", "including", "that", "which",
        "who", "called", "begin", "start", "prescribed", "given", "suspected", "diagnosed", "performed"}


def _bare(w: str) -> str:
    return w.lower().strip(".,;:()")


def answer_span(clean: str, error: str) -> tuple[int, int] | None:
    """Word span [i, j) of `clean` holding the clinical answer that differs from `error`.

    Strip the shared prefix and suffix, then grow the span out to the surrounding
    function words so a partial phrase ("renal" of "renal cell carcinoma") is never cut.
    Returns None when the claims do not share most of their wording.
    """
    a, b = _tok(clean), _tok(error)
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j, k = len(a), len(b)
    while j > i and k > i and a[j - 1] == b[k - 1]:
        j, k = j - 1, k - 1
    if j <= i:
        return None
    while i > 0 and _bare(a[i - 1]) not in FUNC and not a[i - 1].endswith((".", ",", ":")):
        i -= 1
    while j < len(a) and _bare(a[j]) not in FUNC and not a[j - 1].endswith((".", ",", ":")):
        j += 1
    if (j - i) > 0.6 * len(a) or (j - i) > 4:
        return None
    if not any(re.search(r"[A-Za-z]{3,}", w) for w in a[i:j]) or any(_bare(w) in FUNC for w in a[i:j]):
        return None
    return i, j


def _fix_article(words: list[str], at: int) -> None:
    if at > 0 and at < len(words) and words[at - 1].lower() in ("a", "an"):
        art = "an" if words[at][:1].lower() in "aeiou" else "a"
        words[at - 1] = art if words[at - 1].islower() else art.capitalize()


def random_swap(cases, seed: int = 0) -> dict[str, str]:
    """Map case_id -> claim with this patient's answer replaced by another patient's answer."""
    rng = random.Random(seed)
    spans = {}
    for c in cases:
        sp = answer_span(c.clean_claim, c.real_error)
        spans[c.case_id] = sp
    pool: dict[str, list] = {}
    for c in cases:
        pool.setdefault(c.error_type, []).append(c)
    out = {}
    for c in cases:
        donors = [d for d in pool[c.error_type] if d.case_id != c.case_id]
        if not donors:
            continue
        for _ in range(20):
            d = rng.choice(donors)
            sp, dsp = spans[c.case_id], spans[d.case_id]
            if sp and dsp:
                w, dw = _tok(c.clean_claim), _tok(d.clean_claim)
                piece = [x.rstrip(".,;:") for x in dw[dsp[0]: dsp[1]]]
                if w[sp[1] - 1][-1:] in ".,;:":
                    piece[-1] += w[sp[1] - 1][-1]
                if sp[0] == 0:
                    piece[0] = piece[0][:1].upper() + piece[0][1:]
                elif not piece[0].isupper() and piece[0][:1].isupper() and piece[0][1:].islower():
                    piece[0] = piece[0].lower()
                words = w[: sp[0]] + piece + w[sp[1]:]
                _fix_article(words, sp[0])
                new = " ".join(words)
            else:
                new = d.clean_claim  # whole-sentence swap for free-form management/treatment claims
            if new.lower() != c.clean_claim.lower():
                out[c.case_id] = new
                break
    return out
