"""Run reproducible quality and latency evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from bot import classify
from dataset import HOLDOUT_CASES, TEST_CASES


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def evaluate_cases(cases: tuple[tuple[str, str], ...]) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    per_intent_total: Counter[str] = Counter()
    per_intent_correct: Counter[str] = Counter()
    confusion: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for text, expected in cases:
        response = classify(text)
        predicted = response.intent
        correct = predicted == expected
        per_intent_total[expected] += 1
        per_intent_correct[expected] += int(correct)
        confusion[expected][predicted] += 1
        predictions.append(
            {
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "confidence": response.confidence,
            }
        )

    correct_count = sum(item["correct"] for item in predictions)
    total = len(predictions)
    per_intent = {
        intent: {
            "correct": per_intent_correct[intent],
            "total": per_intent_total[intent],
            "accuracy": round(per_intent_correct[intent] / per_intent_total[intent], 4),
        }
        for intent in sorted(per_intent_total)
    }
    return {
        "dataset_size": total,
        "correct": correct_count,
        "accuracy": round(correct_count / total, 4),
        "per_intent": per_intent,
        "confusion": {key: dict(value) for key, value in sorted(confusion.items())},
        "predictions": predictions,
    }


def evaluate() -> dict[str, object]:
    canonical = evaluate_cases(TEST_CASES)
    holdout = evaluate_cases(HOLDOUT_CASES)
    all_cases = TEST_CASES + HOLDOUT_CASES
    overall = evaluate_cases(all_cases)
    latency_ms: list[float] = []
    for _ in range(100):
        for text, _ in all_cases:
            started = perf_counter_ns()
            classify(text)
            latency_ms.append((perf_counter_ns() - started) / 1_000_000)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical": canonical,
        "holdout": holdout,
        "overall": overall,
        "latency_ms": {
            "runs": len(latency_ms),
            "median": round(median(latency_ms), 4),
            "p95": round(percentile(latency_ms, 0.95), 4),
            "max": round(max(latency_ms), 4),
        },
    }


def write_reports(result: dict[str, object]) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    latency = result["latency_ms"]
    canonical = result["canonical"]
    holdout = result["holdout"]
    overall = result["overall"]
    lines = [
        "# Evaluation report",
        "",
        f"- Canonical cases: {canonical['dataset_size']}; accuracy: {canonical['accuracy']:.1%}",
        f"- Holdout cases: {holdout['dataset_size']}; accuracy: {holdout['accuracy']:.1%}",
        f"- Overall cases: {overall['dataset_size']}; accuracy: {overall['accuracy']:.1%}",
        f"- Median classification latency: {latency['median']:.4f} ms",
        f"- P95 classification latency: {latency['p95']:.4f} ms",
        "",
        "## Per-intent accuracy",
        "",
        "| Intent | Correct | Total | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for intent, values in holdout["per_intent"].items():
        lines.append(
            f"| {intent} | {values['correct']} | {values['total']} | {values['accuracy']:.1%} |"
        )
    errors = [item for item in holdout["predictions"] if not item["correct"]]
    lines.extend(["", "## Errors", ""])
    if errors:
        for item in errors:
            lines.append(
                f"- `{item['text']}`: expected `{item['expected']}`, got `{item['predicted']}`"
            )
    else:
        lines.append("No classification errors in the current evaluation set.")
    (ARTIFACTS / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    evaluation = evaluate()
    write_reports(evaluation)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
