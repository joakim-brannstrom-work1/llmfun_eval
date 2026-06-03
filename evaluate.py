"""
evaluate.py - Evaluate LLM strategic decision-making from vision input

Two-stage evaluation:
  Stage 1 (Perception): Object identification accuracy
  Stage 2 (Strategy):  Decision rule application accuracy

Usage:
  python evaluate.py --dataset dataset.json --responses responses.json --report
"""

import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Set
from pathlib import Path


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class GroundTruthSample:
    """One test case with ground truth."""
    sample_id: str
    image_path: str
    objects_present: List[str]
    expected_decision: str
    applicable_rules: List[str]
    notes: str = ""


@dataclass
class LLMResponse:
    """Parsed LLM output for one sample."""
    sample_id: str
    raw_output: str
    identified_objects: List[str]
    decision: str
    reasoning: str = ""


@dataclass
class SampleScore:
    """Per-sample evaluation result."""
    sample_id: str
    perception_iou: float
    strategy_correct: bool
    combined_score: float
    gt_objects: List[str]
    pred_objects: List[str]
    gt_decision: str
    pred_decision: str
    gt_rules: List[str]
    error_type: str = ""


# ============================================================
# METRIC COMPUTATION
# ============================================================

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_perception_metrics(samples, responses):
    tp = fp = fn = 0
    per_type = {}

    for gt, resp in zip(samples, responses):
        gt_set = set(gt.objects_present)
        pred_set = set(resp.identified_objects)

        for obj in gt_set | pred_set:
            if obj not in per_type:
                per_type[obj] = {"tp": 0, "fp": 0, "fn": 0}

            in_gt = obj in gt_set
            in_pred = obj in pred_set

            if in_gt and in_pred:
                tp += 1
                per_type[obj]["tp"] += 1
            elif in_pred and not in_gt:
                fp += 1
                per_type[obj]["fp"] += 1
            elif in_gt and not in_pred:
                fn += 1
                per_type[obj]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    exact_matches = sum(
        1 for gt, resp in zip(samples, responses)
        if set(gt.objects_present) == set(resp.identified_objects)
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match_rate": round(exact_matches / len(samples), 4) if samples else 0.0,
        "per_object_type": {
            obj: {
                "precision": round(c["tp"] / (c["tp"] + c["fp"]), 4) if (c["tp"] + c["fp"]) > 0 else 0.0,
                "recall": round(c["tp"] / (c["tp"] + c["fn"]), 4) if (c["tp"] + c["fn"]) > 0 else 0.0,
            }
            for obj, c in per_type.items()
        },
    }


def compute_strategy_metrics(samples, responses):
    correct = 0
    rule_tp = rule_fp = rule_fn = 0

    for gt, resp in zip(samples, responses):
        if normalize(gt.expected_decision) == normalize(resp.decision):
            correct += 1

        gt_rules = {normalize(r) for r in gt.applicable_rules}
        mentioned = extract_rules(resp.raw_output + " " + resp.reasoning, gt_rules)

        for r in gt_rules:
            if r in mentioned:
                rule_tp += 1
            else:
                rule_fn += 1

        all_mentioned = extract_all_rule_refs(resp.raw_output + " " + resp.reasoning)
        for r in all_mentioned:
            if r not in gt_rules:
                rule_fp += 1

    total = len(samples)
    return {
        "decision_accuracy": round(correct / total, 4) if total else 0.0,
        "rule_recall": round(rule_tp / (rule_tp + rule_fn), 4) if (rule_tp + rule_fn) > 0 else 0.0,
        "rule_precision": round(rule_tp / (rule_tp + rule_fp), 4) if (rule_tp + rule_fp) > 0 else 0.0,
        "hallucinated_rules": rule_fp,
        "missed_rules": rule_fn,
    }


def score_samples(samples, responses, perc_weight=0.4, strat_weight=0.6):
    scores = []
    for gt, resp in zip(samples, responses):
        gt_set = set(gt.objects_present)
        pred_set = set(resp.identified_objects)
        iou = jaccard(gt_set, pred_set)
        correct = normalize(gt.expected_decision) == normalize(resp.decision)

        combined = (iou ** perc_weight) * ((1.0 if correct else 0.0) ** strat_weight)

        perc_fail = iou < 0.5
        strat_fail = not correct
        if perc_fail and strat_fail:
            error = "both"
        elif perc_fail:
            error = "perception_fail"
        elif strat_fail:
            error = "strategy_fail"
        else:
            error = "none"

        scores.append(SampleScore(
            sample_id=gt.sample_id,
            perception_iou=round(iou, 4),
            strategy_correct=correct,
            combined_score=round(combined, 4),
            gt_objects=sorted(gt.objects_present),
            pred_objects=sorted(resp.identified_objects),
            gt_decision=gt.expected_decision,
            pred_decision=resp.decision,
            gt_rules=gt.applicable_rules,
            error_type=error,
        ))
    return scores


# ============================================================
# PARSING HELPERS
# ============================================================

def normalize(text):
    return text.strip().lower().replace("_", " ").replace("-", " ")


def extract_rules(text, known_rules):
    text_lower = normalize(text)
    return {r for r in known_rules if r in text_lower}


def extract_all_rule_refs(text):
    import re
    text_lower = normalize(text)
    patterns = [r'rule[_\s:]+([\w\s]{2,30})']
    found = set()
    for pat in patterns:
        found.update(normalize(m) for m in re.findall(pat, text_lower))
    return found


# ============================================================
# FILE I/O
# ============================================================

def load_dataset(path):
    with open(path) as f:
        data = json.load(f)
    return [GroundTruthSample(**item) for item in data]


def load_responses(path):
    with open(path) as f:
        data = json.load(f)
    return sorted([LLMResponse(**item) for item in data], key=lambda r: r.sample_id)


# ============================================================
# REPORT
# ============================================================

def print_report(results):
    print("\n" + "=" * 64)
    print("  LLM STRATEGIC VISION EVALUATION")
    print("=" * 64)

    s = results["summary"]
    print("\nSamples: {}/{}".format(s["num_evaluated"], s["num_samples"]))
    print("Perception (avg IoU):  {:.4f}".format(s["avg_perception"]))
    print("Strategy (accuracy):   {:.4f}".format(s["avg_strategy"]))
    print("Combined score:        {:.4f}".format(s["avg_combined"]))

    p = results["perception"]
    print("\n--- Perception Metrics ---")
    print("  Object Precision:     {:.4f}".format(p["precision"]))
    print("  Object Recall:        {:.4f}".format(p["recall"]))
    print("  Object F1:            {:.4f}".format(p["f1"]))
    print("  Exact Match Rate:     {:.4f}".format(p["exact_match_rate"]))

    if p.get("per_object_type"):
        print("\n  Per-Object Breakdown:")
        for obj, m in sorted(p["per_object_type"].items()):
            print("    {:25s}  P={:.3f}  R={:.3f}".format(obj, m["precision"], m["recall"]))

    st = results["strategy"]
    print("\n--- Strategy Metrics ---")
    print("  Decision Accuracy:    {:.4f}".format(st["decision_accuracy"]))
    print("  Rule Recall:          {:.4f}".format(st["rule_recall"]))
    print("  Rule Precision:       {:.4f}".format(st["rule_precision"]))
    print("  Hallucinated Rules:   {}".format(st["hallucinated_rules"]))
    print("  Missed Rules:         {}".format(st["missed_rules"]))

    error_counts = {}
    for sc in results["per_sample"]:
        e = sc["error_type"]
        error_counts[e] = error_counts.get(e, 0) + 1

    if any(v > 0 for k, v in error_counts.items() if k != "none"):
        print("\n--- Error Breakdown ---")
        for etype in ["perception_fail", "strategy_fail", "both", "none"]:
            if error_counts.get(etype, 0) > 0:
                print("  {:20s}: {}".format(etype, error_counts[etype]))

    failures = [s for s in results["per_sample"] if s["error_type"] != "none"]
    if failures:
        print("\n--- Top Failures ---")
        for f in sorted(failures, key=lambda x: x["combined_score"])[:5]:
            print("\n  [{}] error={}".format(f["sample_id"], f["error_type"]))
            print("    Objects:  GT={}".format(f["gt_objects"]))
            print("             Pred={}".format(f["pred_objects"]))
            print("    Decision: GT={}".format(f["gt_decision"]))
            print("             Pred={}".format(f["pred_decision"]))

    print("\n" + "=" * 64)


# ============================================================
# MAIN
# ============================================================

def evaluate(dataset_path, responses_path, output_path, show_report):
    samples = load_dataset(dataset_path)
    responses = load_responses(responses_path)

    resp_map = {r.sample_id: r for r in responses}
    aligned = [(s, resp_map[s.sample_id]) for s in samples if s.sample_id in resp_map]
    samples_aligned, responses_aligned = list(zip(*aligned)) if aligned else ([], [])

    perception = compute_perception_metrics(samples_aligned, responses_aligned)
    strategy = compute_strategy_metrics(samples_aligned, responses_aligned)
    sample_scores = score_samples(samples_aligned, responses_aligned)

    avg_perc = sum(s.perception_iou for s in sample_scores) / len(sample_scores) if sample_scores else 0
    avg_strat = sum(1 for s in sample_scores if s.strategy_correct) / len(sample_scores) if sample_scores else 0
    avg_combined = sum(s.combined_score for s in sample_scores) / len(sample_scores) if sample_scores else 0

    results = {
        "summary": {
            "num_samples": len(samples),
            "num_evaluated": len(samples_aligned),
            "avg_perception": round(avg_perc, 4),
            "avg_strategy": round(avg_strat, 4),
            "avg_combined": round(avg_combined, 4),
        },
        "perception": perception,
        "strategy": strategy,
        "per_sample": [asdict(s) for s in sample_scores],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to {}".format(output_path))

    if show_report:
        print_report(results)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM strategic vision decisions")
    parser.add_argument("--dataset", required=True, help="Ground truth dataset JSON")
    parser.add_argument("--responses", required=True, help="LLM responses JSON")
    parser.add_argument("--output", default="evaluation_results.json", help="Output JSON path")
    parser.add_argument("--report", action="store_true", help="Print human-readable report")
    args = parser.parse_args()

    evaluate(args.dataset, args.responses, args.output, args.report)


if __name__ == "__main__":
    main()
