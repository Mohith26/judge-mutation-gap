"""Score every case under clean / negation / random_swap / real conditions. Resumable."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .data import load_cases
from .perturb import negate, random_swap

SPLITS = {
    "val": "data/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv",
    "test": "data/MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv",
}


def build_items(cases, conditions):
    swaps = random_swap(cases, seed=0)
    for c in cases:
        claims = {"clean": c.clean_claim, "negation": negate(c.clean_claim), "real": c.real_error}
        if c.case_id in swaps:
            claims["random_swap"] = swaps[c.case_id]
        for cond in conditions:
            if cond in claims:
                yield c, cond, claims[cond]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["nli", "llm"], required=True)
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--conditions", default="clean,negation,random_swap,real")
    ap.add_argument("--limit", type=int, default=0, help="first N cases only")
    ap.add_argument("--max-seconds", type=int, default=540, help="stop early; rerun to resume")
    a = ap.parse_args(argv)

    from .judges import LLMJudge, NLIJudge
    judge = NLIJudge() if a.judge == "nli" else LLMJudge()
    cases = load_cases(SPLITS[a.split])
    if a.limit:
        cases = cases[: a.limit]
    out = Path(f"results/raw_{a.judge}_{a.split}.jsonl")
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["case_id"], r["condition"]))
    t0, n = time.time(), 0
    with out.open("a") as f:
        for c, cond, claim in build_items(cases, a.conditions.split(",")):
            if (c.case_id, cond) in done:
                continue
            ps, pn, pa = judge(c.evidence, claim)
            f.write(json.dumps({"judge": judge.name, "split": a.split, "case_id": c.case_id,
                                "error_type": c.error_type, "condition": cond, "claim": claim,
                                "p_supported": ps, "p_not_supported": pn, "p_not_addressed": pa}) + "\n")
            f.flush()
            n += 1
            if time.time() - t0 > a.max_seconds:
                print(f"time budget hit after {n} items; rerun to resume")
                return 2
    print(f"done, {n} new items -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
