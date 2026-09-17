"""
src/eval/score_fullchain.py

Full-chain scoring: runs handle_query() on all 21 gold pairs and scores
each against its bucket's own pass rule. Reported as three separate
pass rates (A/A2, B, C), never blended -- the errors are asymmetric
(a fabricated answer on a bucket-B pair is much worse than a correct
decline), so one accuracy number would hide the thing that matters most.
"""

import json
from src.agents.dispatcher import handle_query

with open("data/eval/qa_pairs.json", encoding="utf-8") as f:
    gold_pairs = json.load(f)

BUCKET_A = {"g001","g002","g003","g004","g005","g006","g007",
            "g009","g010","g011","g012","g020","g021"}
BUCKET_A2 = {"g013"}
BUCKET_B = {"g008","g014","g015","g016","g019"}
BUCKET_C = {"g017","g018"}


def score_A(pair, result):
    if result["status"] != "answered":
        return False, "wrong status (expected answered)"
    if result["grounded"] is not True:
        return False, "not grounded"
    if set(result["citations"]) != set(pair["expected_citations"]):
        return False, f"citation mismatch: expected {pair['expected_citations']}, got {result['citations']}"
    return True, ""


def score_A2(pair, result):
    if result["status"] != "answered":
        return False, "wrong status (expected answered)"
    if result["grounded"] is not True:
        return False, "not grounded"
    if not set(pair["expected_citations"]).issubset(set(result["citations"])):
        return False, f"expected citations not covered: expected {pair['expected_citations']}, got {result['citations']}"
    return True, ""


def score_B(pair, result):
    if result["status"] == "answered":
        return False, "FABRICATION: answered a pair expected to be declined"
    if result["status"] in ("ungrounded", "low_confidence_retrieval"):
        return True, ""
    return False, f"unexpected status: {result['status']}"


def score_C(pair, result):
    if result["status"] == "out_of_scope":
        return True, ""
    return False, f"unexpected status: {result['status']}"


bucket_results = {"A": [], "A2": [], "B": [], "C": []}

for pair in gold_pairs:
    pid = pair["id"]
    result = handle_query(pair["query"])

    if pid in BUCKET_A:
        passed, reason = score_A(pair, result)
        bucket_results["A"].append((pid, passed, reason))
    elif pid in BUCKET_A2:
        passed, reason = score_A2(pair, result)
        bucket_results["A2"].append((pid, passed, reason))
    elif pid in BUCKET_B:
        passed, reason = score_B(pair, result)
        bucket_results["B"].append((pid, passed, reason))
    elif pid in BUCKET_C:
        passed, reason = score_C(pair, result)
        bucket_results["C"].append((pid, passed, reason))

    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {pid}  status={result['status']}  {reason}")

print("\n" + "=" * 50)
for bucket_name, items in bucket_results.items():
    if not items:
        continue
    n_pass = sum(1 for _, p, _ in items if p)
    print(f"Bucket {bucket_name}: {n_pass}/{len(items)} passed")