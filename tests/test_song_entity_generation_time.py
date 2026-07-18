from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_song_entity_generation_time_prompts import (  # noqa: E402
    build_prompt_rows,
    load_jsonl,
    validate_contexts,
    validate_controls,
)
from run_song_entity_generation_time_probe import (  # noqa: E402
    parse_playlist,
    render_verification_prompt,
)
from verify_song_entity_catalog import (  # noqa: E402
    apple_url,
    candidate_evidence,
    classify_catalog_evidence,
    musicbrainz_url,
    normalize_name,
)


class ExportTests(unittest.TestCase):
    def test_specs_have_expected_shape(self) -> None:
        contexts = load_jsonl(
            PROJECT_ROOT / "data" / "qwen_scope_song_entity_generation_time_specs.jsonl"
        )
        controls = load_jsonl(
            PROJECT_ROOT / "data" / "qwen_scope_song_entity_pair_controls.jsonl"
        )
        validate_contexts(contexts)
        validate_controls(controls)
        template = (
            PROJECT_ROOT / "prompts" / "song_entity_generation_time_prompt.md"
        ).read_text(encoding="utf-8")
        rows = build_prompt_rows(contexts, template, [17, 29, 43])
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["generation_id"] for row in rows}), 12)


class ParserTests(unittest.TestCase):
    def test_parses_prompt_tail_and_numbered_blocks(self) -> None:
        text = (
            "Playlist:\n1. Title:City Lights\n"
            "   Artist: Night Driver\n"
            "   Reason: restrained beat for a rainy road\n"
            "2. Title: Quiet Reflection\n"
            "Artist: Soft Circuit\n"
            "Reason: low-distraction instrumental texture\n"
        )
        rows = parse_playlist(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["title"], "City Lights")
        self.assertEqual(rows[1]["artist"], "Soft Circuit")
        start, end = rows[0]["title_span"]
        self.assertEqual(text[start:end], "City Lights")

    def test_drops_incomplete_block(self) -> None:
        rows = parse_playlist("1. Title: Incomplete\nArtist: Nobody\n")
        self.assertEqual(rows, [])

    def test_parses_pending_and_markdown_fields(self) -> None:
        text = (
            "Playlist:\n1. **Title**:\nNeon Memory\n"
            "**Artist**: Night Index\n"
            "**Reason**: restrained synth motion\n"
        )
        rows = parse_playlist(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "Neon Memory")
        start, end = rows[0]["title_span"]
        self.assertEqual(text[start:end], "Neon Memory")

    def test_parses_inline_fields_and_strips_title_quotes(self) -> None:
        text = (
            'Playlist:\n1. Title: "Rainy Days" Artist: The City Pop Band '
            'Reason: calm urban mood for a rainy road\n'
            '2. Title: "Urban Chill" Artist: Indie Pop Group '
            'Reason: restrained low-pressure beat\n'
        )
        rows = parse_playlist(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["title"], "Rainy Days")
        self.assertEqual(rows[0]["artist"], "The City Pop Band")
        start, end = rows[0]["title_span"]
        self.assertEqual(text[start:end], "Rainy Days")

    def test_parses_title_by_artist_with_separate_reason(self) -> None:
        text = (
            'Playlist:\n1. Title: "Midnight Rain" by DJ Shadow\n'
            'Reason: slow ambient beat for a rainy night\n\n'
            '2. Title: "Electric Feel" - The Chemical Brothers\n'
            'Reason: pulsating high-energy drums\n'
        )
        rows = parse_playlist(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["title"], "Midnight Rain")
        self.assertEqual(rows[0]["artist"], "DJ Shadow")
        self.assertEqual(rows[1]["artist"], "The Chemical Brothers")

    def test_parses_recommendation_section_for_missing_inline_reasons(self) -> None:
        text = (
            'Playlist:\n1. Title: "City Lights" Artist: Björk\n'
            '2. Title: "Night Moves" Artist: Massive Attack\n\n'
            'Recommendations:\n'
            '1. "City Lights" by Björk creates a calm urban mood.\n'
            '2. "Night Moves" by Massive Attack has a restrained beat.\n'
        )
        rows = parse_playlist(text)
        self.assertEqual(len(rows), 2)
        self.assertIn("calm urban mood", rows[0]["reason"])
        self.assertEqual(rows[1]["artist"], "Massive Attack")

    def test_verification_option_mapping_survives_order_flip(self) -> None:
        template = (
            PROJECT_ROOT / "prompts" / "song_entity_pair_verification_prompt.md"
        ).read_text(encoding="utf-8")
        prompt, mapping = render_verification_prompt(
            template,
            title="稻香",
            artist="周杰伦",
            source_condition="self_attributed",
            options=[("unknown", "Unknown"), ("known_exact", "Known exact pair")],
        )
        self.assertEqual(mapping, {"A": "unknown", "B": "known_exact"})
        self.assertIn("Earlier you generated", prompt)
        self.assertIn("稻香", prompt)


class CatalogTests(unittest.TestCase):
    def test_normalization_handles_width_case_and_punctuation(self) -> None:
        self.assertEqual(normalize_name("Ｂad-Guy!"), normalize_name("bad guy"))

    def test_exact_and_conflict_evidence(self) -> None:
        candidates = [
            {"title": "Blinding Lights", "artist": "The Weeknd", "source_id": "1"},
            {"title": "Blinding Lights", "artist": "Billie Eilish", "source_id": "2"},
        ]
        exact = candidate_evidence("Blinding Lights", "The Weeknd", candidates)
        conflict = candidate_evidence("Blinding Lights", "Queen", candidates)
        self.assertEqual(len(exact["exact"]), 1)
        self.assertEqual(len(conflict["exact"]), 0)
        self.assertEqual(len(conflict["title_matches"]), 2)

    def test_artist_alias_can_confirm_exact_pair(self) -> None:
        candidates = [
            {"title": "青花瓷", "artist": "周杰倫", "source_id": "1"},
            {"title": "泡沫", "artist": "G.E.M.", "source_id": "2"},
        ]
        jay = candidate_evidence(
            "青花瓷",
            "周杰伦",
            candidates,
            accepted_artists=["周杰倫", "Jay Chou"],
        )
        gem = candidate_evidence(
            "泡沫",
            "邓紫棋",
            candidates,
            accepted_artists=["G.E.M.", "鄧紫棋"],
        )
        self.assertEqual(len(jay["exact"]), 1)
        self.assertEqual(len(gem["exact"]), 1)

    def test_catalog_queries_recall_by_title_before_artist_comparison(self) -> None:
        musicbrainz = musicbrainz_url("Blinding Lights", "Wrong Artist")
        apple = apple_url("Blinding Lights", "Wrong Artist")
        self.assertIn("Blinding", musicbrainz)
        self.assertIn("Blinding", apple)
        self.assertNotIn("Wrong", musicbrainz)
        self.assertNotIn("Wrong", apple)
        self.assertIn(
            "Wrong",
            musicbrainz_url("Blinding Lights", "Wrong Artist", targeted=True),
        )
        self.assertIn(
            "Wrong",
            apple_url("Blinding Lights", "Wrong Artist", targeted=True),
        )

    def test_catalog_labels_require_both_sources_for_unverified(self) -> None:
        empty = {"exact": [], "title_matches": []}
        conflict = {"exact": [], "title_matches": [{"title": "x"}]}
        self.assertEqual(
            classify_catalog_evidence(
                [
                    {"status": "ok", "evidence": empty},
                    {"status": "ok", "evidence": empty},
                ]
            ),
            "unverified",
        )
        self.assertEqual(
            classify_catalog_evidence(
                [
                    {"status": "ok", "evidence": conflict},
                    {"status": "ok", "evidence": empty},
                ]
            ),
            "catalog_conflict",
        )
        self.assertEqual(
            classify_catalog_evidence(
                [
                    {"status": "error", "evidence": empty},
                    {"status": "ok", "evidence": empty},
                ]
            ),
            "verification_error",
        )


if __name__ == "__main__":
    unittest.main()
