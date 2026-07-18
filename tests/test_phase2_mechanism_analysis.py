from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from phase2_mechanism_analysis import (  # noqa: E402
    classify_template,
    cohens_kappa,
    confirmatory_mechanism_label,
    h1_metrics,
    h2_interaction,
    median_layer_shifts,
    normalize_entity,
    parse_candidate_free_response,
    patch_relation_shift,
    score_candidate_free_condition,
    select_hash_ordered_controls,
)


class Phase2MechanismAnalysisTests(unittest.TestCase):
    def test_patch_relation_shift_matches_frozen_contrast(self) -> None:
        self.assertAlmostEqual(patch_relation_shift(-1.0, -2.0, -2.5, -2.0), 1.5)

    def test_template_label_uses_strict_threshold_and_two_layers(self) -> None:
        self.assertEqual(
            classify_template({20: 0.051, 28: 0.08, 36: -0.2}), "prior_masked"
        )
        self.assertEqual(
            classify_template({20: -0.051, 28: -0.08, 36: 0.01}), "wrong_binding"
        )
        self.assertEqual(
            classify_template({20: 0.05, 28: 0.2, 36: 0.0}), "indeterminate"
        )

    def test_medians_are_taken_across_neutral_titles(self) -> None:
        medians = median_layer_shifts({20: [0.1, 0.3, -2.0], 28: [0.2, 0.4, 0.3]})
        self.assertEqual(medians, {20: 0.1, 28: 0.3})

    def test_confirmatory_label_requires_non_indeterminate_agreement(self) -> None:
        self.assertEqual(
            confirmatory_mechanism_label(["prior_masked", "prior_masked"]),
            "prior_masked",
        )
        self.assertEqual(
            confirmatory_mechanism_label(["prior_masked", "wrong_binding"]),
            "indeterminate",
        )
        self.assertEqual(
            confirmatory_mechanism_label(["indeterminate", "indeterminate"]),
            "indeterminate",
        )

    def test_kappa_includes_indeterminate_as_third_class(self) -> None:
        first = ["prior_masked", "wrong_binding", "indeterminate"] * 2
        self.assertEqual(cohens_kappa(first, first), 1.0)

    def test_degenerate_single_label_kappa_fails_conservatively(self) -> None:
        labels = ["indeterminate"] * 5
        self.assertEqual(cohens_kappa(labels, labels), 0.0)

    def test_h1_technical_failures_remain_in_denominator(self) -> None:
        rows = [
            {"template_labels": ["prior_masked", "prior_masked"]},
            {"template_labels": ["wrong_binding", "wrong_binding"]},
            {
                "template_labels": ["prior_masked", "wrong_binding"],
                "technical_failure": True,
            },
        ]
        result = h1_metrics(
            rows,
            minimum_raw_agreement=1.0,
            minimum_kappa=0.0,
            minimum_classifiable_coverage=0.5,
        )
        self.assertEqual(result["cluster_count"], 3)
        self.assertEqual(result["raw_three_class_agreement"], 1.0)
        self.assertAlmostEqual(result["same_non_indeterminate_coverage"], 2 / 3)
        self.assertTrue(result["passes"])

    def test_candidate_free_parser_is_strict(self) -> None:
        self.assertEqual(
            parse_candidate_free_response("Artist: Bon Iver")["normalized_artist"],
            "boniver",
        )
        self.assertEqual(parse_candidate_free_response(" ABSTAIN ")["status"], "abstained")
        self.assertEqual(
            parse_candidate_free_response("I think it is Artist: Bon Iver")["status"],
            "invalid",
        )
        self.assertEqual(
            parse_candidate_free_response("Artist: Bon Iver\nThanks")["status"],
            "invalid",
        )
        self.assertEqual(
            parse_candidate_free_response("Artist: Bon Iver Artist: Sam Smith")[
                "status"
            ],
            "invalid",
        )

    def test_two_paraphrases_are_scored_independently_then_averaged(self) -> None:
        result = score_candidate_free_condition(
            ["Artist: Bon Iver", "Artist: Sam Smith"], "Bon Iver"
        )
        self.assertEqual(result["indicators"], [1, 0])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["continued_error_count"], 1)
        self.assertTrue(result["paraphrase_disagreement"])

    def test_h2_uses_stratified_cluster_bootstrap(self) -> None:
        rows = []
        for index in range(15):
            rows.append(
                {
                    "title": f"Prior {index}",
                    "normalized_title": f"prior{index}",
                    "mechanism_label": "prior_masked",
                    "naive_accuracy": 0.0,
                    "anti_prior_accuracy": 1.0,
                }
            )
            rows.append(
                {
                    "title": f"Wrong {index}",
                    "normalized_title": f"wrong{index}",
                    "mechanism_label": "wrong_binding",
                    "naive_accuracy": 0.5,
                    "anti_prior_accuracy": 0.5,
                }
            )
        result = h2_interaction(rows, bootstrap_samples=100)
        self.assertEqual(result["interaction"], 1.0)
        self.assertEqual(result["bootstrap_percentile_95_interval"], [1.0, 1.0])
        self.assertTrue(result["passes"])

    def test_h2_rejects_duplicate_normalized_titles(self) -> None:
        rows = [
            {
                "title": "A",
                "normalized_title": "a",
                "mechanism_label": "prior_masked",
                "naive_accuracy": 0.0,
                "anti_prior_accuracy": 1.0,
            },
            {
                "title": "A!",
                "normalized_title": "a",
                "mechanism_label": "wrong_binding",
                "naive_accuracy": 0.0,
                "anti_prior_accuracy": 0.0,
            },
        ]
        with self.assertRaisesRegex(ValueError, "unique normalized-title"):
            h2_interaction(rows, bootstrap_samples=10, minimum_clusters_per_class=1)

    def test_h2_rejects_non_frozen_fractional_scores(self) -> None:
        rows = [
            {
                "title": "A",
                "mechanism_label": "prior_masked",
                "naive_accuracy": 0.25,
                "anti_prior_accuracy": 1.0,
            },
            {
                "title": "B",
                "mechanism_label": "wrong_binding",
                "naive_accuracy": 0.0,
                "anti_prior_accuracy": 0.0,
            },
        ]
        with self.assertRaisesRegex(ValueError, "0, 0.5, or 1"):
            h2_interaction(rows, bootstrap_samples=10, minimum_clusters_per_class=1)

    def test_normalization_matches_catalog_style(self) -> None:
        self.assertEqual(normalize_entity("Beyonce\u0301 & JAY-Z"), "beyonc\u00e9jayz")

    def test_control_order_is_deterministic_and_excludes_conflict(self) -> None:
        first = select_hash_ordered_controls("conflict", ["b", "a", "conflict", "a"])
        second = select_hash_ordered_controls("conflict", ["a", "b"])
        self.assertEqual(first, second)
        self.assertNotIn("conflict", first)


if __name__ == "__main__":
    unittest.main()
