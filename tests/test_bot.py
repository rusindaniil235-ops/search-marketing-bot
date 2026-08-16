import json
import unittest

from bot import INTENTS, available_topics, classify, normalize_text
from dataset import HOLDOUT_CASES, TEST_CASES


class BotTestCase(unittest.TestCase):
    def test_evaluation_dataset(self) -> None:
        errors = []
        for text, expected in TEST_CASES:
            predicted = classify(text).intent
            if predicted != expected:
                errors.append((text, expected, predicted))
        self.assertEqual(errors, [])

    def test_normalization(self) -> None:
        self.assertEqual(normalize_text("  КАЧЕСТВО, данных! Ёлка "), "качество данных елка")

    def test_empty_input_uses_fallback(self) -> None:
        response = classify("   ")
        self.assertEqual(response.intent, "fallback")
        self.assertEqual(response.confidence, 0.0)

    def test_response_is_json_serializable(self) -> None:
        payload = classify("Что такое атрибуция?").to_dict()
        json.dumps(payload, ensure_ascii=False)

    def test_intent_names_are_unique(self) -> None:
        names = [intent.name for intent in INTENTS]
        self.assertEqual(len(names), len(set(names)))

    def test_topics_match_intents(self) -> None:
        self.assertEqual(len(available_topics()), len(INTENTS))

    def test_holdout_accuracy_is_acceptable(self) -> None:
        correct = sum(classify(text).intent == expected for text, expected in HOLDOUT_CASES)
        self.assertGreaterEqual(correct / len(HOLDOUT_CASES), 0.70)


if __name__ == "__main__":
    unittest.main()
