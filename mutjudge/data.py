"""Load MEDEC-MS rows into (evidence, clean claim, real error) cases.

Each MEDEC error row has a clinician-written wrong sentence (Error Sentence) and its
correct version (Corrected Sentence). We treat the rest of the corrected note as the
patient record (evidence) and the target sentence as the claim to verify, so the clean
and perturbed runs share exactly the same evidence, as in the Forus design.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    case_id: str
    error_type: str
    evidence: str
    clean_claim: str
    real_error: str


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def evidence_without(text: str, sentence: str) -> str:
    """Remove the target sentence from the note; raise if it is not present."""
    t, s = _norm(text), _norm(sentence)
    if s not in t:
        raise ValueError("target sentence not found in note")
    return _norm(t.replace(s, " ", 1))


def load_cases(path: str | Path) -> list[Case]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Error Flag") != "1":
                continue
            clean, err = _norm(row["Corrected Sentence"]), _norm(row["Error Sentence"])
            if not clean or not err or clean == err:
                continue
            ev = evidence_without(row["Corrected Text"], row["Corrected Sentence"])
            cases.append(Case(row["Text ID"], row["Error Type"], ev, clean, err))
    return cases
