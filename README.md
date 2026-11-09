# judge-mutation-gap

Mutation testing for claim-level clinical evidence judges, on public data: how much do synthetic
perturbations overstate a judge's ability to catch realistic clinical errors?

This reproduces the evaluation design from Forus's engineering post
["How we evaluate our LLM judge: a perturbation-based approach"](https://build.forus.com/how-we-evaluate-our-llm-judge-a-perturbation-based-approach)
(Isabel Chien, June 2026) with open data and open models, and measures the open problem the post
ends on: the gap between synthetic errors and real ones.

## Setup

The post's design: take answers you know are correct, run the judge on them (clean run, gives the
false-positive rate), then perturb some of them into plausible-but-wrong answers on the *same*
patient record (perturbed run, gives the detection rate). The judge returns one of three verdicts:
Supported, Not Supported, Not Addressed, and only Not Supported flags.

Forus's gold PA data is private, so this uses **MEDEC-MS** (Ben Abacha et al., ACL Findings 2025,
CC BY 4.0), the public part of the MEDIQA-CORR benchmark. Each error case is a clinical vignette in
which one sentence (a diagnosis, management step, treatment, drug or causal organism) was replaced by
a medical annotator with a wrong but plausible alternative. That gives, per patient:

- **evidence**: the rest of the note, identical across all runs
- **clean claim**: the correct sentence
- **real error**: the clinician-written wrong sentence (a plausible distractor chosen for this patient)

and two synthetic families generated from the clean claim:

- **negation**: rule-based polarity flip ("Rotavirus is suspected." -> "Rotavirus is not suspected.")
- **random swap**: this patient's answer replaced by another patient's answer of the same error type
  ("...diagnosed with pleural effusion" -> "...diagnosed with alpha-1 antitrypsin deficiency"). This is the
  "error incoherent with the rest of the chart" the post warns is too easy.

All 311 error cases of the MEDEC-MS test split, 4 claims each, 1,244 judge calls per judge.

Judges (both local, CPU, no API):

- `Qwen2.5-1.5B-Instruct` prompted with the post's three verdict definitions; verdict probabilities
  from the next-token logits of the option letters.
- `DeBERTa-v3-base-mnli-fever-anli` NLI cross-encoder; entailment / contradiction / neutral map onto
  Supported / Not Supported / Not Addressed.

The headline metric is the one the post anchors on: fix the threshold so the clean run has a 10%
false-positive rate, then read detection per family at that threshold. AUROC (clean vs family) and a
paired metric (share of patients where the wrong claim is not scored as more suspicious than the
correct one) are reported too. 95% CIs by bootstrap.

## Results (MEDEC-MS test, 311 patients)

Qwen2.5-1.5B-Instruct judge, Forus-style verdicts (Not Addressed does not flag):

| perturbation family | argmax "detection" | AUROC vs clean | detection at 10% clean FP (95% CI) | wrong claim not scored as more suspicious |
|---|---|---|---|---|
| negation | 89.7% | 0.556 | 3.9% (1.9-6.4%) | 41.2% |
| random swap | 94.2% | 0.778 | **45.3%** (40.2-51.1%) | 18.3% |
| real clinician-written error | 69.8% | 0.528 | **10.6%** (7.4-14.1%) | 38.9% |

At its natural argmax decision the same judge flags **67.5% of the correct claims**.

DeBERTa NLI judge:

| mode | clean FP (argmax) | real-error detection at 10% FP | AUROC real vs clean |
|---|---|---|---|
| Not Addressed does not flag (Forus design) | 7.1% | 8.7% | 0.493 |
| Not Addressed collapses into Not Supported | **98.4%** | 15.1% | 0.521 |

Full tables, including per-error-type breakdowns, are in [results/RESULTS.md](results/RESULTS.md);
every per-claim score is in `results/raw_*_test.jsonl`.

## What this shows

1. **Context-free swaps overstate real detection by about 4x.** At a matched 10% clean
   false-positive rate the LLM judge catches 45% of random-swap mutations but only 11% of the
   clinician-written errors on the same patients, which is roughly the chance level (10%). A
   mutation suite built from swaps alone would have reported a judge that works.
2. **Raw detection numbers without the clean run are meaningless.** Argmax "detection" is 70-94% for
   every family only because the judge flags two-thirds of correct claims. This is the post's reason
   for anchoring on the clean false-positive rate first, measured.
3. **Rule-based negation is not even an easy family here.** The 1.5B judge barely separates
   negated claims from correct ones (AUROC 0.56); naive perturbations test robustness to surface
   form, not clinical judgment, as the post says.
4. **The Not Addressed verdict is what keeps an evidence judge usable.** For the NLI judge, almost
   every conclusion in a vignette is "not entailed" by the findings (diagnoses need inference), so
   collapsing silence into Not Supported raises the clean false-positive rate from 7.1% to 98.4%.

## What this is not

- MEDEC vignettes are MedQA-style cases, not prior authorization forms, and the claims are single
  sentences, not multi-part PA answers. No atomic decomposition or retrieval stage is used: each
  note fits in the judge's context.
- Both judges are small local models. The absolute numbers say nothing about a production judge;
  the point is the *relative* gap between families measured with the same judge on the same patients.
  The harness takes any `judge(evidence, claim) -> (p_supported, p_not_supported, p_not_addressed)`.
- The swap generator is heuristic. In an audit of 40 random swaps (`results/swap_audit_sample.txt`)
  2 were ungrammatical, which if anything makes swaps easier to flag. 184 of 311 cases get an
  in-sentence span swap; the rest (free-form management/treatment sentences) get the donor's whole
  sentence.
- Single run, deterministic scoring (no sampling). Model revisions are pinned in `mutjudge/judges.py`.

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/get_data.sh                                   # MEDEC-MS at a pinned commit, sha256-checked
.venv/bin/python -m pytest -q                         # 15 tests, no model downloads
.venv/bin/python -m mutjudge.run --judge nli --split test   # ~4 min on Apple Silicon CPU
.venv/bin/python -m mutjudge.run --judge llm --split test   # ~16 min; resumable, rerun to continue
.venv/bin/python -m mutjudge.report                   # writes results/RESULTS.md and summary.json
```

Hardware for the reported run: Apple Silicon arm64, CPU only, torch 2.15.0, transformers 5.22.2.

## Next steps that would make this more useful

- Context-anchored synthetic perturbations (an LLM picks a wrong answer coherent with the chart, as in
  the post) scored against the clinician-written errors, to see how much of the 4x gap they close.
- Direct versus consistency-based detection on multi-sentence answers.

## Citation

MEDEC: Ben Abacha, Yim, Fu, Sun, Yetisgen, Xia, Lin. "MEDEC: A Benchmark for Medical Error Detection
and Correction in Clinical Notes." Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.1159/
