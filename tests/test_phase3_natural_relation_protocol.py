from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Phase3NaturalRelationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (
                PROJECT_ROOT
                / "config"
                / "phase3_natural_relation_discovery_protocol.json"
            ).read_text(encoding="utf-8")
        )

    def test_generation_counts_and_splits_are_frozen(self) -> None:
        generation = self.protocol["generation"]
        pilot = len(generation["discovery_contexts"]) * len(
            generation["pilot_seeds"]
        )
        full_discovery = len(generation["discovery_contexts"]) * (
            len(generation["pilot_seeds"]) + len(generation["expansion_seeds"])
        )
        holdout = len(generation["holdout_contexts"]) * len(
            generation["holdout_seeds"]
        )
        self.assertEqual(pilot, generation["pilot_playlist_count"])
        self.assertEqual(
            full_discovery, generation["maximum_discovery_playlist_count"]
        )
        self.assertEqual(holdout, generation["holdout_playlist_count"])

        discovery_seeds = set(generation["pilot_seeds"]) | set(
            generation["expansion_seeds"]
        )
        self.assertTrue(discovery_seeds.isdisjoint(generation["holdout_seeds"]))
        discovery_contexts = {
            row["context_id"] for row in generation["discovery_contexts"]
        }
        holdout_contexts = {
            row["context_id"] for row in generation["holdout_contexts"]
        }
        self.assertTrue(discovery_contexts.isdisjoint(holdout_contexts))

    def test_recoverability_precedes_probe_and_intervention(self) -> None:
        self.assertTrue(
            self.protocol["catalog_labeling"][
                "generic_title_collision_is_not_a_known_relation_error"
            ]
        )
        self.assertEqual(
            self.protocol["knowledge_recoverability"]["eligible_label"],
            "recoverable_relation_conflict",
        )
        self.assertEqual(
            self.protocol["probe_development"]["primary_position"],
            "title_end_pre_artist",
        )
        self.assertIn(
            "holdout pre-artist hidden-state AUROC exceeds the best frozen surface baseline",
            self.protocol["mechanism_gate"]["required_before_intervention"],
        )

    def test_holdout_is_prompt_and_entity_disjoint(self) -> None:
        freeze = self.protocol["holdout_freeze"]
        self.assertIn("pipeline freeze receipt committed", freeze["do_not_generate_until"])
        self.assertIn("prompt-template wording disjoint", freeze["disjointness"])
        self.assertIn("normalized title disjoint", freeze["disjointness"])
        self.assertIn(
            "normalized emitted and reference artist disjoint",
            freeze["disjointness"],
        )

    def test_stress_yield_cannot_be_reported_as_prevalence(self) -> None:
        boundaries = self.protocol["claim_boundaries"]
        self.assertTrue(any("cannot estimate" in boundary for boundary in boundaries))
        self.assertIn("reason-first output order", self.protocol["excluded_primary_factors"])


if __name__ == "__main__":
    unittest.main()
