from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_song_entity_component_patching_probe import (  # noqa: E402
    SELECTED_LAYERS,
    build_bundle,
    select_smoke_controls as select_export_smoke_controls,
)
from export_song_entity_relation_binding_probe import load_jsonl  # noqa: E402
from run_song_entity_component_patching_probe import (  # noqa: E402
    component_recovery,
    overlapping_positions,
    select_smoke_controls as select_run_smoke_controls,
)


class ComponentPatchingExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = load_jsonl(
            PROJECT_ROOT
            / "data"
            / "qwen_scope_song_entity_relation_binding_controls.jsonl"
        )

    def test_smoke_reuses_complete_swap_blocks(self) -> None:
        export_rows = select_export_smoke_controls(self.rows)
        run_rows = select_run_smoke_controls(self.rows)
        self.assertEqual(export_rows, run_rows)
        self.assertEqual(len(run_rows), 12)
        for block_id in {row["block_id"] for row in run_rows}:
            self.assertEqual(
                sum(row["block_id"] == block_id for row in run_rows),
                2,
            )

    def test_bundle_registers_selected_components_and_endpoint(self) -> None:
        bundle = build_bundle(self.rows)
        self.assertEqual(bundle["selected_layers"], SELECTED_LAYERS)
        self.assertEqual(bundle["analysis_layers"], [14, 16, 18, 21, 24, 27])
        self.assertEqual(
            bundle["components"],
            ["attention", "mlp", "full_residual"],
        )
        self.assertEqual(bundle["endpoint_tolerance"], 0.02)


class ComponentPatchingRunnerTests(unittest.TestCase):
    def test_recovery_is_signed_toward_neutral(self) -> None:
        self.assertAlmostEqual(component_recovery(5.0, 1.0, 3.0), 0.5)
        self.assertAlmostEqual(component_recovery(1.0, 4.0, 3.0), 2.0 / 3.0)

    def test_recovery_ignores_tiny_title_effect(self) -> None:
        self.assertIsNone(component_recovery(1.02, 1.0, 1.01))

    def test_overlap_ignores_special_token_offsets(self) -> None:
        self.assertEqual(
            overlapping_positions([(0, 0), (0, 4), (4, 8)], [4, 8]),
            [2],
        )


if __name__ == "__main__":
    unittest.main()
