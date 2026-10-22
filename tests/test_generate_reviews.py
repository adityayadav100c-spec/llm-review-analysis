import unittest

from generate_reviews import THEMES, generate_rows


class SyntheticReviewTests(unittest.TestCase):
    def test_generates_balanced_deterministic_labeled_examples(self):
        first = generate_rows(2, seed=17)
        second = generate_rows(2, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(THEMES) * 2 * 2)
        self.assertEqual({row["theme"] for row in first}, set(THEMES))
        self.assertEqual({row["sentiment"] for row in first}, {"positive", "negative"})
        self.assertTrue(all(row["is_synthetic"] == "true" for row in first))

    def test_rejects_empty_group_count(self):
        with self.assertRaises(ValueError):
            generate_rows(0, seed=42)


if __name__ == "__main__":
    unittest.main()
