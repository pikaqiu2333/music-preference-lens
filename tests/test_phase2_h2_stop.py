from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import phase2_mechanism_analysis as analysis  # noqa: E402


def make_h2_rows(prior_count: int, wrong_count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, count, naive, anti_prior in (
        ("prior_masked", prior_count, 0.0, 1.0),
        ("wrong_binding", wrong_count, 0.5, 0.5),
    ):
        for index in range(count):
            rows.append(
                {
                    "title": f"{label} {index}",
                    "normalized_title": f"{label}{index}",
                    "mechanism_label": label,
                    "naive_accuracy": naive,
                    "anti_prior_accuracy": anti_prior,
                }
            )
    return rows


class H2MinimumClassStopTests(unittest.TestCase):
    def assert_underpowered_stop(
        self, prior_count: int, wrong_count: int
    ) -> dict[str, object]:
        with patch.object(
            analysis.random,
            "Random",
            side_effect=AssertionError("underpowered H2 must not bootstrap"),
        ):
            result = analysis.h2_interaction(
                make_h2_rows(prior_count, wrong_count), bootstrap_samples=10
            )

        self.assertEqual(
            result["class_counts"],
            {"prior_masked": prior_count, "wrong_binding": wrong_count},
        )
        self.assertFalse(result["estimable"])
        self.assertFalse(result["minimum_class_size_gate"])
        self.assertFalse(result["passes"])
        self.assertNotIn("interaction", result)
        self.assertNotIn("bootstrap_samples", result)
        self.assertNotIn("bootstrap_percentile_95_interval", result)
        return result

    def test_14_and_15_clusters_stops_without_bootstrap(self) -> None:
        result = self.assert_underpowered_stop(14, 15)
        self.assertIn("at least 15", result["reason"])

    def test_0_and_15_clusters_stops_without_bootstrap(self) -> None:
        result = self.assert_underpowered_stop(0, 15)
        self.assertIn("at least 15", result["reason"])

    def test_15_and_15_clusters_runs_confirmatory_bootstrap(self) -> None:
        result = analysis.h2_interaction(make_h2_rows(15, 15), bootstrap_samples=10)

        self.assertTrue(result["estimable"])
        self.assertTrue(result["minimum_class_size_gate"])
        self.assertEqual(result["interaction"], 1.0)
        self.assertEqual(result["bootstrap_percentile_95_interval"], [1.0, 1.0])
        self.assertTrue(result["passes"])


if __name__ == "__main__":
    unittest.main()
