import csv
import json

import pytest

from mutjudge import report
from mutjudge.data import evidence_without, load_cases
from mutjudge.metrics import auroc, bootstrap_ci, rate, threshold_at_fpr
from mutjudge.perturb import answer_span, negate, random_swap

HDR = ["Text ID", "Text", "Sentences", "Error Flag", "Error Type", "Error Sentence ID",
       "Error Sentence", "Corrected Sentence", "Corrected Text"]


def _write(tmp_path, rows):
    p = tmp_path / "medec.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, HDR)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _row(i, typ, ctx, good, bad):
    return {"Text ID": f"t{i}", "Text": f"{ctx} {bad}", "Sentences": "", "Error Flag": "1",
            "Error Type": typ, "Error Sentence ID": "1", "Error Sentence": bad,
            "Corrected Sentence": good, "Corrected Text": f"{ctx} {good}"}


@pytest.fixture
def cases(tmp_path):
    rows = [
        _row(0, "diagnosis", "Rib notching on X-ray.", "Patient is diagnosed with coarctation of the aorta.",
             "Patient is diagnosed with aortic stenosis."),
        _row(1, "diagnosis", "Apical lung mass.", "Patient is diagnosed with a pancoast tumor.",
             "Patient is diagnosed with a pulmonary hamartoma."),
        _row(2, "diagnosis", "Hip pain in a boy.", "Legg-Calve-Perthes disease is suspected after examination.",
             "Slipped capital femoral epiphysis is suspected after examination."),
        _row(3, "management", "Acute cauda equina signs.", "The patient is immediately transferred to the emergency department.",
             "Non-emergent inpatient spinal manipulation program is advised."),
        _row(4, "management", "Burn on the chest.", "Escharotomy is performed.", "Barium XR is ordered."),
        {**_row(5, "NA", "Normal note.", "", ""), "Error Flag": "0"},
    ]
    return load_cases(_write(tmp_path, rows))


def test_load_keeps_only_error_rows_and_strips_claim_from_evidence(cases):
    assert [c.case_id for c in cases] == ["t0", "t1", "t2", "t3", "t4"]
    c = cases[0]
    assert c.evidence == "Rib notching on X-ray."
    assert c.clean_claim not in c.evidence and c.real_error not in c.evidence


def test_evidence_without_raises_when_sentence_missing():
    with pytest.raises(ValueError):
        evidence_without("A. B.", "C.")


@pytest.mark.parametrize("claim,expected", [
    ("Rotavirus is suspected.", "Rotavirus is not suspected."),
    ("Patient was given aspirin.", "Patient was not given aspirin."),
    ("Escharotomy performed.", "It is not the case that escharotomy performed."),
    ("Doctor prescribed methimazole.", "Doctor not prescribed methimazole."),
])
def test_negate(claim, expected):
    assert negate(claim) == expected


def test_answer_span_grows_to_phrase_boundaries():
    clean = "Patient is diagnosed with renal cell carcinoma after CT."
    err = "Patient is diagnosed with renal oncocytoma after CT."
    i, j = answer_span(clean, err)
    assert clean.split()[i:j] == ["renal", "cell", "carcinoma"]


def test_answer_span_none_for_rewritten_sentence():
    assert answer_span("Escharotomy is performed.", "Barium XR is ordered.") is None


def test_random_swap_uses_other_patient_same_type_and_is_deterministic(cases):
    a, b = random_swap(cases, seed=0), random_swap(cases, seed=0)
    assert a == b
    by = {c.case_id: c for c in cases}
    for cid, new in a.items():
        assert new.lower() != by[cid].clean_claim.lower()
    # span swap keeps this patient's context and fixes the article
    assert a["t1"].startswith("Patient is diagnosed with ") and "pancoast" not in a["t1"]
    # management claims are rewritten sentences, so the swap is the donor's whole claim
    assert a["t3"] == by["t4"].clean_claim and a["t4"] == by["t3"].clean_claim


def test_article_is_fixed_after_span_swap(tmp_path):
    rows = [
        _row(0, "diagnosis", "Barrel chest.", "Patient is diagnosed with a pleural effusion.",
             "Patient is diagnosed with a pneumothorax."),
        _row(1, "diagnosis", "Low AAT.", "Patient is diagnosed with alpha-1 deficiency.",
             "Patient is diagnosed with asthma."),
    ]
    cs = load_cases(_write(tmp_path, rows))
    assert random_swap(cs)["t0"] == "Patient is diagnosed with an alpha-1 deficiency."


def test_auroc_known_values():
    assert auroc([0, 0], [1, 1]) == 1.0
    assert auroc([1, 1], [0, 0]) == 0.0
    assert auroc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert auroc([0.1, 0.4], [0.35, 0.8]) == 0.75


def test_threshold_controls_false_positive_rate():
    clean = [i / 100 for i in range(100)]
    t = threshold_at_fpr(clean, 0.10)
    assert rate(clean, t) == pytest.approx(0.09, abs=0.011)
    assert rate([t + 1e-9], t) == 1.0


def test_bootstrap_ci_brackets_mean():
    lo, hi = bootstrap_ci([1, 0] * 50)
    assert lo < 0.5 < hi


def test_report_summary_on_synthetic_scores(tmp_path):
    rows = []
    for i in range(50):
        for cond, pns in (("clean", i / 100), ("negation", 0.9), ("random_swap", 0.9), ("real", i / 100)):
            rows.append({"judge": "stub", "case_id": f"c{i}", "error_type": "diagnosis", "condition": cond,
                         "p_supported": 1 - pns, "p_not_supported": pns, "p_not_addressed": 0.0})
    p = tmp_path / "raw_stub_test.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    s = report.summarize(p, fpr=0.10)
    fam = s["modes"]["lenient"]["families"]
    assert fam["random_swap"]["auroc_vs_clean"] == 1.0 and fam["random_swap"]["detection_at_fpr"] == 1.0
    assert fam["real"]["auroc_vs_clean"] == 0.5 and fam["real"]["paired_not_higher_than_clean"] == 1.0


def test_runner_is_resumable_with_stub_judge(tmp_path, monkeypatch, cases):
    from mutjudge import judges, run
    calls = []

    class Stub:
        name = "stub"

        def __init__(self, *a, **k):
            pass

        def __call__(self, ev, claim):
            calls.append(claim)
            return (0.2, 0.7, 0.1)

    monkeypatch.setattr(judges, "NLIJudge", Stub)
    monkeypatch.setattr(run, "load_cases", lambda _: cases)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    assert run.main(["--judge", "nli", "--split", "test"]) == 0
    n = len(calls)
    assert n == 5 * 4
    assert run.main(["--judge", "nli", "--split", "test"]) == 0
    assert len(calls) == n  # nothing re-scored
