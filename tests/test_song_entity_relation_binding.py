from __future__ import annotations

import sys
import unittest
from importlib.util import find_spec
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_song_entity_relation_binding_probe import (  # noqa: E402
    build_catalog_precheck_rows,
    build_bundle,
    load_jsonl,
    select_smoke_controls as select_export_smoke_controls,
    validate_controls,
)
from run_song_entity_relation_binding_probe import (  # noqa: E402
    build_likelihood_prefix,
    build_sae_text,
    cross_validate_paired_features,
    make_relation_folds,
    render_choice_prompt,
    select_smoke_controls as select_run_smoke_controls,
)


class RelationBindingExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = load_jsonl(
            PROJECT_ROOT
            / "data"
            / "qwen_scope_song_entity_relation_binding_controls.jsonl"
        )

    def test_controls_are_balanced_derangements(self) -> None:
        validate_controls(self.rows)
        self.assertEqual(len(self.rows), 20)
        for language in ("en", "zh"):
            subset = [row for row in self.rows if row["language"] == language]
            self.assertEqual(
                Counter(row["correct_artist"].casefold() for row in subset),
                Counter(row["wrong_artist"].casefold() for row in subset),
            )
        blocks = {row["block_id"] for row in self.rows}
        self.assertEqual(len(blocks), 10)
        for block_id in blocks:
            block = [row for row in self.rows if row["block_id"] == block_id]
            self.assertEqual(len(block), 2)
            self.assertEqual(
                Counter(row["correct_artist"].casefold() for row in block),
                Counter(row["wrong_artist"].casefold() for row in block),
            )

    def test_smoke_subset_preserves_artist_marginals(self) -> None:
        export_rows = select_export_smoke_controls(self.rows)
        run_rows = select_run_smoke_controls(self.rows)
        self.assertEqual(export_rows, run_rows)
        self.assertEqual(len(run_rows), 12)
        self.assertEqual(
            Counter(row["correct_artist"].casefold() for row in run_rows),
            Counter(row["wrong_artist"].casefold() for row in run_rows),
        )

    def test_bundle_registers_primary_gate(self) -> None:
        bundle = build_bundle(self.rows)
        self.assertEqual(bundle["interpretation_threshold"], 0.80)
        self.assertEqual(bundle["feature_count"], 32)
        self.assertEqual(bundle["pair_folds"], 5)

    def test_catalog_precheck_has_matched_exact_and_mismatch_rows(self) -> None:
        checks = build_catalog_precheck_rows(self.rows)
        self.assertEqual(len(checks), 40)
        self.assertEqual(Counter(row["group"] for row in checks), {
            "known_exact": 20,
            "artist_mismatch": 20,
        })
        exact = next(row for row in checks if row["pair_id"] == "rel_qing_hua_ci__exact")
        mismatch = next(
            row for row in checks if row["pair_id"] == "rel_qing_hua_ci__mismatch"
        )
        self.assertIn("Jay Chou", exact["accepted_artists"])
        self.assertNotIn("accepted_artists", mismatch)


class RelationBindingRunnerTests(unittest.TestCase):
    def test_sae_spans_target_artist_and_relation_marker(self) -> None:
        template = "Catalog fact check.\nTitle: {title}\nArtist: {artist}\nRelation:"
        text, artist_span, relation_span = build_sae_text(
            template,
            "Shape of You",
            "Ed Sheeran",
        )
        self.assertEqual(text[artist_span[0] : artist_span[1]], "Ed Sheeran")
        self.assertEqual(text[relation_span[0] : relation_span[1]], "Relation")

    def test_prompt_rendering_keeps_title_fixed(self) -> None:
        likelihood = build_likelihood_prefix(
            "Title: {title}\nArtist: ",
            "稻香",
        )
        choice = render_choice_prompt(
            'Title "{title}"\nA. {artist_a}\nB. {artist_b}\nAnswer:',
            "稻香",
            "周杰伦",
            "林俊杰",
        )
        self.assertEqual(likelihood, "Title: 稻香\nArtist: ")
        self.assertIn('Title "稻香"', choice)
        self.assertIn("A. 周杰伦", choice)

    def test_folds_hold_out_complete_relations_and_both_languages(self) -> None:
        rows = load_jsonl(
            PROJECT_ROOT
            / "data"
            / "qwen_scope_song_entity_relation_binding_controls.jsonl"
        )
        smoke = select_run_smoke_controls(rows)
        folds = make_relation_folds(smoke, 5)
        self.assertEqual(len(folds), 3)
        self.assertEqual(set().union(*folds), {row["relation_id"] for row in smoke})
        for fold in folds:
            languages = {
                row["language"] for row in smoke if row["relation_id"] in fold
            }
            self.assertEqual(languages, {"en", "zh"})
            held_blocks = {
                row["block_id"] for row in smoke if row["relation_id"] in fold
            }
            for block_id in held_blocks:
                block_ids = {
                    row["relation_id"]
                    for row in smoke
                    if row["block_id"] == block_id
                }
                self.assertTrue(block_ids.issubset(fold))

    @unittest.skipUnless(find_spec("numpy"), "numpy is provided by the HF Job runtime")
    def test_paired_cv_recovers_shared_exact_direction(self) -> None:
        import numpy as np

        rows = []
        for language in ("en", "zh"):
            for index in range(10):
                rows.append(
                    {
                        "relation_id": f"{language}_{index}",
                        "block_id": f"{language}_block_{index // 2}",
                        "language": language,
                        "exact_pair_end": np.asarray([2.0, 0.0, 1.0], dtype=np.float32),
                        "mismatch_pair_end": np.asarray(
                            [0.0, 0.0, 1.0], dtype=np.float32
                        ),
                    }
                )
        result = cross_validate_paired_features(rows, "pair_end", 2, 5)
        self.assertEqual(result["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
