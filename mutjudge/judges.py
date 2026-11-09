"""Two local claim-level judges with the Forus three-way verdict.

Both return probabilities over (supported, not_supported, not_addressed).
- NLIJudge: an off-the-shelf NLI cross-encoder; entailment / contradiction / neutral map
  directly onto Supported / Not Supported / Not Addressed.
- LLMJudge: a small instruct LLM prompted with the three verdict definitions from the
  Forus post; probabilities come from the next-token logits of the option letters.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

torch.set_num_threads(4)


class NLIJudge:
    name = "deberta-v3-base-nli"

    def __init__(self, model_id: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        rev = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=rev).eval()
        lab = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.idx = (lab["entailment"], lab["contradiction"], lab["neutral"])

    @torch.no_grad()
    def __call__(self, evidence: str, claim: str) -> tuple[float, float, float]:
        enc = self.tok(evidence, claim, truncation="only_first", max_length=512, return_tensors="pt")
        p = torch.softmax(self.model(**enc).logits[0], -1)
        return tuple(float(p[i]) for i in self.idx)


PROMPT = """You verify one claim from a prior authorization answer against a patient's clinical record.

Clinical record:
{evidence}

Claim: {claim}

Choose one verdict:
A. Supported: the record affirms the claim.
B. Not Supported: the claim conflicts with the record, through direct contradiction or because the record points to something else.
C. Not Addressed: the record is silent on the claim.

Answer with a single letter."""


class LLMJudge:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.name = model_id.split("/")[-1]
        rev = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, dtype=torch.float32).eval()
        self.ids = [self.tok.encode(x, add_special_tokens=False)[0] for x in ("A", "B", "C")]

    @torch.no_grad()
    def __call__(self, evidence: str, claim: str) -> tuple[float, float, float]:
        msgs = [{"role": "user", "content": PROMPT.format(evidence=evidence, claim=claim)}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt")
        logits = self.model(**enc).logits[0, -1].float()
        p = torch.softmax(logits[self.ids], -1)
        return float(p[0]), float(p[1]), float(p[2])
