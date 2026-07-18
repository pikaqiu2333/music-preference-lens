from __future__ import annotations

import base64
import json
import sys
import unittest
import zlib
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_holdout_generations import analyze  # noqa: E402
from build_embedded_hf_job import embed_bundle  # noqa: E402
from decode_zlib_job_artifact import decode_artifact  # noqa: E402
from export_holdout_generation_probe import (  # noqa: E402
    DISCOVERY_SEEDS,
    HOLDOUT_SEEDS,
    build_bundle,
)
from export_independent_holdout_verifier_probe import (  # noqa: E402
    attach_references,
    build_bundle as build_verifier_bundle,
    load_json,
    load_jsonl,
    select_holdout_rows,
    selection_hash,
)
from export_holdout_title_contrast_probe import (  # noqa: E402
    build_bundle as build_title_contrast_bundle,
)
from export_holdout_sequence_causal_trace import (  # noqa: E402
    build_bundle as build_sequence_trace_bundle,
)
from extract_job_log_artifacts import extract_chunks, extract_plain  # noqa: E402
from run_independent_holdout_verifier_probe import (  # noqa: E402
    binary_metrics,
    combined_prediction,
    confirmation_passes,
)
from run_holdout_title_contrast_probe import (  # noqa: E402
    diagnostic_class,
    summarize_path,
)
from run_holdout_sequence_causal_trace import (  # noqa: E402
    earliest_sustained_layer,
    encode_artifact_chunks,
    normalized_recovery,
)


class IndependentHoldoutExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = json.loads(
            (
                PROJECT_ROOT
                / "runs"
                / "qwen_scope_song_entity_generation_time_bundle.json"
            ).read_text(encoding="utf-8")
        )

    def test_holdout_seeds_are_new_and_generation_count_is_frozen(self) -> None:
        bundle = build_bundle(self.source)
        self.assertFalse(DISCOVERY_SEEDS & set(HOLDOUT_SEEDS))
        self.assertEqual(bundle["seeds"], [59, 71, 83, 101, 127])
        self.assertEqual(bundle["expected_generation_count"], 20)
        self.assertEqual(len(bundle["contexts"]), 4)

    def test_generation_settings_match_discovery_protocol(self) -> None:
        bundle = build_bundle(self.source)
        self.assertEqual(bundle["generation"]["temperature"], 0.7)
        self.assertEqual(bundle["generation"]["top_p"], 0.9)
        self.assertEqual(bundle["generation"]["max_new_tokens"], 384)

    def test_standalone_job_embedding_replaces_only_assignment(self) -> None:
        placeholder = "__BUNDLE__"
        runner = f'VALUE = "{placeholder}"\nif VALUE == "{placeholder}":\n    raise ValueError\n'
        embedded = embed_bundle(runner, b'{"ok":true}', placeholder, "zlib")
        self.assertEqual(embedded.count(placeholder), 1)
        self.assertNotIn(f'VALUE = "{placeholder}"', embedded)


class IndependentHoldoutAnalyzerTests(unittest.TestCase):
    def test_analyzer_uses_frozen_parser_on_complete_block(self) -> None:
        source = {
            "contexts": [
                {
                    "context_id": "test",
                    "generation_prompt": "Playlist:\n1. Title:\n",
                }
            ],
            "seeds": [59],
            "minimum_parsed_generation_count": 1,
            "minimum_parsed_pair_count": 1,
        }
        raw = [
            {
                "generation_id": "test__seed59",
                "context_id": "test",
                "seed": 59,
                "completion": (
                    ' "Ocean Eyes"\nArtist: Billie Eilish\n'
                    "Reason: A quiet emotional song."
                ),
            }
        ]
        rows, summary = analyze(source, raw)
        self.assertTrue(summary["technical_gate"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "Ocean Eyes")
        self.assertEqual(rows[0]["artist"], "Billie Eilish")

    def test_analyzer_rejects_missing_generation_ids(self) -> None:
        source = {
            "contexts": [{"context_id": "test", "generation_prompt": "Playlist:"}],
            "seeds": [59, 71],
            "minimum_parsed_generation_count": 1,
            "minimum_parsed_pair_count": 1,
        }
        with self.assertRaises(ValueError):
            analyze(
                source,
                [
                    {
                        "generation_id": "test__seed59",
                        "context_id": "test",
                        "seed": 59,
                        "completion": "x",
                    }
                ],
            )


class IndependentHoldoutVerifierExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog_rows = load_jsonl(
            PROJECT_ROOT / "runs" / "independent_holdout_catalog_verified.jsonl"
        )
        self.generation_bundle = load_json(
            PROJECT_ROOT / "runs" / "independent_holdout_generation_bundle.json"
        )
        self.raw_generations = load_jsonl(
            PROJECT_ROOT / "runs" / "independent_holdout_raw_generations.jsonl"
        )

    def test_hash_selection_is_frozen_balanced_and_large_enough(self) -> None:
        selected, metadata = select_holdout_rows(self.catalog_rows)
        self.assertEqual(
            Counter(row["catalog_label"] for row in selected),
            Counter({"verified_exact": 9, "catalog_conflict": 9}),
        )
        self.assertEqual(metadata["double_source_exact_pool_count"], 9)
        self.assertEqual(metadata["unique_shared_conflict_pool_count"], 21)
        self.assertEqual(metadata["selected_per_label"], 9)
        for label in ("verified_exact", "catalog_conflict"):
            hashes = [
                row["selection_hash"]
                for row in selected
                if row["catalog_label"] == label
            ]
            self.assertEqual(hashes, sorted(hashes))
            self.assertTrue(
                all(row["selection_hash"] == selection_hash(row) for row in selected)
            )

    def test_references_are_independent_of_scores_and_differ_from_emitted(self) -> None:
        selected, _ = select_holdout_rows(self.catalog_rows)
        attached = attach_references(selected)
        for row in attached:
            self.assertNotEqual(
                row["artist"].casefold(), row["reference_artist"].casefold()
            )
            self.assertNotIn("margin", row)
            self.assertNotIn("score", row)

    def test_bundle_reconstructs_all_prefixes_and_uses_only_holdout_seeds(self) -> None:
        bundle = build_verifier_bundle(
            self.catalog_rows, self.generation_bundle, self.raw_generations
        )
        self.assertEqual(len(bundle["records"]), 18)
        self.assertEqual(bundle["frozen_rule"]["required_correct_per_label"], 7)
        self.assertFalse(
            DISCOVERY_SEEDS & {row["seed"] for row in bundle["records"]}
        )
        for row in bundle["records"]:
            self.assertTrue(row["generation_prefix"].strip())
            self.assertFalse(
                row["generation_prefix"].endswith(row["emitted_artist"])
            )
            self.assertTrue(row["relation_cluster_id"])


class IndependentHoldoutVerifierRunnerTests(unittest.TestCase):
    def test_combined_rule_is_the_frozen_zero_threshold_or(self) -> None:
        self.assertFalse(combined_prediction(0.1, 0.2))
        self.assertTrue(combined_prediction(-0.1, 0.2))
        self.assertTrue(combined_prediction(0.1, -0.2))
        self.assertFalse(combined_prediction(0.0, 0.0))

    def test_confirmation_requires_each_prespecified_gate(self) -> None:
        rows = []
        for index in range(9):
            rows.append(
                {
                    "catalog_label": "verified_exact",
                    "warning": index >= 7,
                }
            )
            rows.append(
                {
                    "catalog_label": "catalog_conflict",
                    "warning": index < 7,
                }
            )
        metrics = binary_metrics(rows, "warning")
        rule = {
            "minimum_events_per_label": 8,
            "minimum_balanced_accuracy": 0.75,
            "minimum_exact_specificity": 0.75,
            "minimum_conflict_sensitivity": 0.75,
        }
        self.assertTrue(confirmation_passes(metrics, rule))
        self.assertFalse(confirmation_passes(metrics, rule, technical_gate=False))
        self.assertFalse(
            confirmation_passes(
                {**metrics, "conflict_sensitivity": 6 / 9}, rule
            )
        )


class HoldoutTitleContrastExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier_bundle = load_json(
            PROJECT_ROOT / "runs" / "independent_holdout_verifier_bundle.json"
        )
        self.verifier_rows = load_jsonl(
            PROJECT_ROOT / "runs" / "independent_holdout_verifier_rows.jsonl"
        )

    def test_exact_duplicate_candidate_event_is_removed_but_controls_remain(self) -> None:
        bundle = build_title_contrast_bundle(
            self.verifier_bundle, self.verifier_rows
        )
        self.assertEqual(len(bundle["records"]), 17)
        self.assertEqual(
            bundle["focus_role_counts"],
            {
                "exact_clean_control": 6,
                "sequence_exact_false_positive": 3,
                "choice_conflict_hit": 5,
                "choice_conflict_miss": 3,
            },
        )

    def test_control_titles_are_distinct_and_not_known_for_either_candidate(self) -> None:
        bundle = build_title_contrast_bundle(
            self.verifier_bundle, self.verifier_rows
        )
        for row in bundle["records"]:
            self.assertEqual(len(row["control_titles"]), 2)
            self.assertEqual(len({item["title"] for item in row["control_titles"]}), 2)
            self.assertNotIn(
                row["title"].casefold(),
                {item["title"].casefold() for item in row["control_titles"]},
            )
            candidate_artists = {
                row["emitted_artist"].casefold(),
                row["reference_artist"].casefold(),
            }
            self.assertTrue(
                all(
                    item["known_true_artist"].casefold() not in candidate_artists
                    for item in row["control_titles"]
                )
            )


class HoldoutTitleContrastRunnerTests(unittest.TestCase):
    def test_diagnostic_class_separates_prior_masking_from_missing_relation(self) -> None:
        self.assertEqual(
            diagnostic_class("catalog_conflict", 0.4, -0.3),
            "latent_relation_masked_by_prior",
        )
        self.assertEqual(
            diagnostic_class("catalog_conflict", 0.4, 0.1),
            "relation_not_recovered",
        )
        self.assertEqual(
            diagnostic_class("verified_exact", 0.4, 0.1),
            "correct_and_relation_specific",
        )
        self.assertEqual(
            diagnostic_class("verified_exact", 0.4, -0.1),
            "correct_without_relation_contrast",
        )

    def test_summary_keeps_empty_label_subgroups_explicit(self) -> None:
        rows = [
            {
                "catalog_label": "catalog_conflict",
                "paths": {
                    "choice": {
                        "signed_relation_delta": 0.5,
                        "absolute_correct": True,
                        "delta_direction_correct": True,
                        "diagnostic_class": "correct_and_relation_specific",
                    }
                },
            }
        ]
        summary = summarize_path(rows, "choice")
        self.assertEqual(summary["verified_exact"]["count"], 0)
        self.assertIsNone(
            summary["verified_exact"]["mean_signed_relation_delta"]
        )


class HoldoutSequenceCausalTraceExportTests(unittest.TestCase):
    def test_bundle_covers_all_events_layers_and_components(self) -> None:
        title_bundle = load_json(
            PROJECT_ROOT / "runs" / "holdout_title_contrast_bundle.json"
        )
        title_rows = load_jsonl(
            PROJECT_ROOT / "runs" / "holdout_title_contrast_rows.jsonl"
        )
        bundle = build_sequence_trace_bundle(title_bundle, title_rows)
        self.assertEqual(len(bundle["records"]), 17)
        self.assertEqual(bundle["full_residual_layers"], list(range(1, 29)))
        self.assertEqual(
            {(item["layer"], item["component"]) for item in bundle["component_interventions"]},
            {
                (layer, component)
                for layer in (18, 21, 24, 27)
                for component in ("attention", "mlp")
            },
        )
        self.assertTrue(
            all(
                row["choice_diagnostic_class"]
                and row["sequence_diagnostic_class"]
                for row in bundle["records"]
            )
        )


class HoldoutSequenceCausalTraceRunnerTests(unittest.TestCase):
    def test_recovery_is_relative_to_each_factual_control_effect(self) -> None:
        self.assertAlmostEqual(normalized_recovery(2.0, 0.0, 1.0, 0.1), 0.5)
        self.assertAlmostEqual(normalized_recovery(-2.0, 0.0, -1.0, 0.1), 0.5)
        self.assertIsNone(normalized_recovery(0.05, 0.0, 0.03, 0.1))

    def test_earliest_sustained_layer_requires_consecutive_points(self) -> None:
        points = [
            {"layer": 1, "mean_recovery": 0.6, "toward_source_rate": 0.8},
            {"layer": 2, "mean_recovery": 0.4, "toward_source_rate": 0.9},
            {"layer": 3, "mean_recovery": 0.7, "toward_source_rate": 0.8},
            {"layer": 4, "mean_recovery": 0.8, "toward_source_rate": 0.9},
        ]
        self.assertEqual(earliest_sustained_layer(points), 3)

    def test_chunked_artifact_round_trips_below_log_line_limit(self) -> None:
        source = {"rows": [{"value": "x" * 1000}], "summary": {"ok": True}}
        chunks = encode_artifact_chunks(source, maximum_chars=80)
        self.assertTrue(all(len(chunk) <= 80 for chunk in chunks))
        decoded = json.loads(
            zlib.decompress(base64.b64decode("".join(chunks))).decode("utf-8")
        )
        self.assertEqual(decoded, source)
        self.assertEqual(decode_artifact("".join(chunks)), source)
        chunk_log = "\n".join(
            "CHUNK="
            + json.dumps({"index": index, "total": len(chunks), "data": data})
            for index, data in enumerate(chunks)
        )
        self.assertEqual(extract_chunks(chunk_log, "CHUNK="), source)

    def test_plain_log_extractor_requires_one_summary(self) -> None:
        text = 'ROW={"id":1}\nROW={"id":2}\nSUMMARY={"ok":true}\n'
        self.assertEqual(
            extract_plain(text, "ROW=", "SUMMARY="),
            {"rows": [{"id": 1}, {"id": 2}], "summary": {"ok": True}},
        )


if __name__ == "__main__":
    unittest.main()
