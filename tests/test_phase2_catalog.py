from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_phase2_catalog import (  # noqa: E402
    CATALOG_VERIFIER_VERSION,
    REFERENCE_SEMANTICS,
    CatalogRequestSession,
    append_jsonl,
    apple_url,
    apply_title_cluster_summaries,
    classify_phase2_catalog_evidence,
    musicbrainz_url,
    normalize_name,
    read_jsonl,
    run_cli,
    verify_pair,
    write_jsonl,
)
from verify_song_entity_catalog import (  # noqa: E402
    apple_url as original_apple_url,
    musicbrainz_url as original_musicbrainz_url,
    normalize_name as original_normalize_name,
)


FIXED_TIME = "2026-07-13T12:34:56.000000Z"


def musicbrainz_payload(
    pairs: list[tuple[str, str, str]],
) -> dict[str, Any]:
    return {
        "recordings": [
            {
                "id": source_id,
                "title": title,
                "artist-credit": [{"name": artist}],
            }
            for title, artist, source_id in pairs
        ]
    }


def apple_payload(pairs: list[tuple[str, str, str]]) -> dict[str, Any]:
    return {
        "results": [
            {
                "wrapperType": "track",
                "kind": "song",
                "trackId": source_id,
                "trackName": title,
                "artistName": artist,
            }
            for title, artist, source_id in pairs
        ]
    }


def response(
    payload: dict[str, Any] | None = None,
    *,
    raw_body: str | None = None,
    status: int = 200,
    error: str | None = None,
) -> dict[str, Any]:
    if raw_body is None:
        raw_body = json.dumps(payload or {}, separators=(",", ":"))
    return {
        "http_status": status,
        "raw_response_body": raw_body,
        "error": error,
    }


class FakeFetcher:
    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def __call__(self, url: str) -> Any:
        self.calls.append(url)
        if url not in self.routes:
            raise AssertionError(f"unexpected URL: {url}")
        value = self.routes[url]
        if isinstance(value, list):
            if not value:
                raise AssertionError(f"no response remains for URL: {url}")
            value = value.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def catalog_routes(
    title: str,
    emitted_artist: str,
    musicbrainz_pairs: list[tuple[str, str, str]],
    apple_pairs: list[tuple[str, str, str]],
    *,
    musicbrainz_targeted: Any | None = None,
    apple_targeted: Any | None = None,
) -> dict[str, Any]:
    routes = {
        musicbrainz_url(title, emitted_artist): response(
            musicbrainz_payload(musicbrainz_pairs)
        ),
        apple_url(title, emitted_artist): response(apple_payload(apple_pairs)),
    }
    emitted_key = normalize_name(emitted_artist)
    if not any(
        normalize_name(pair_title) == normalize_name(title)
        and normalize_name(pair_artist) == emitted_key
        for pair_title, pair_artist, _ in musicbrainz_pairs
    ):
        routes[musicbrainz_url(title, emitted_artist, targeted=True)] = (
            musicbrainz_targeted
            if musicbrainz_targeted is not None
            else response(musicbrainz_payload([]))
        )
    if not any(
        normalize_name(pair_title) == normalize_name(title)
        and normalize_name(pair_artist) == emitted_key
        for pair_title, pair_artist, _ in apple_pairs
    ):
        routes[apple_url(title, emitted_artist, targeted=True)] = (
            apple_targeted
            if apple_targeted is not None
            else response(apple_payload([]))
        )
    return routes


def offline_verify(
    title: str,
    artist: str,
    routes: dict[str, Any],
    *,
    accepted_artists: list[str] | None = None,
    max_attempts: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]], FakeFetcher]:
    archive: list[dict[str, Any]] = []
    fetcher = FakeFetcher(routes)
    result = verify_pair(
        title,
        artist,
        accepted_artists=accepted_artists,
        fetcher=fetcher,
        archive=archive,
        now=lambda: FIXED_TIME,
        sleep_seconds=0.0,
        max_attempts=max_attempts,
        sleep_fn=lambda _: None,
    )
    return result, archive, fetcher


class StrictClassificationTests(unittest.TestCase):
    def test_reuses_frozen_url_and_normalization_semantics(self) -> None:
        self.assertIs(normalize_name, original_normalize_name)
        self.assertEqual(
            musicbrainz_url("Song", "Artist"),
            original_musicbrainz_url("Song", "Artist"),
        )
        self.assertEqual(
            apple_url("Song", "Artist"),
            original_apple_url("Song", "Artist"),
        )
        self.assertEqual(
            normalize_name("\uff22\uff41\uff44-Guy!"),
            normalize_name("bad guy"),
        )

    def test_strict_exact_requires_emitted_artist_from_both_sources(self) -> None:
        title = "Exact Song"
        artist = "Exact Artist"
        routes = catalog_routes(
            title,
            artist,
            [(title, artist, "mb-exact"), (title, "Other", "mb-other")],
            [(title, "exact-artist", "101"), (title, "Other", "102")],
        )
        result, _, fetcher = offline_verify(title, artist, routes)
        self.assertEqual(result["catalog_label"], "strict_exact")
        self.assertTrue(result["confirmatory_catalog_eligible"])
        self.assertEqual(result["classification_reason"], "emitted_artist_supported_by_both")
        self.assertEqual(len(fetcher.calls), 2)

    def test_one_source_emitted_support_is_excluded(self) -> None:
        title = "One Source Song"
        artist = "Emitted Artist"
        routes = catalog_routes(
            title,
            artist,
            [(title, artist, "mb-emitted")],
            [(title, "Reference Artist", "201")],
        )
        result, _, _ = offline_verify(title, artist, routes)
        self.assertEqual(result["catalog_label"], "excluded")
        self.assertEqual(
            result["classification_reason"],
            "emitted_artist_supported_by_one_source_only",
        )
        self.assertFalse(result["confirmatory_catalog_eligible"])
        self.assertIsNone(result["catalog_reference"])

    def test_unique_shared_non_emitted_artist_is_strict_conflict(self) -> None:
        title = "Conflict Song"
        artist = "Wrong Artist"
        routes = catalog_routes(
            title,
            artist,
            [(title, "The Correct Artist", "mb-reference")],
            [("conflict-song", "the correct artist", "301")],
        )
        result, archive, _ = offline_verify(title, artist, routes)
        self.assertEqual(result["catalog_label"], "strict_conflict")
        self.assertTrue(result["confirmatory_catalog_eligible"])
        self.assertEqual(result["reference_artist"], "The Correct Artist")
        self.assertEqual(
            result["normalized_reference_artist"], "thecorrectartist"
        )
        self.assertEqual(result["reference_semantics"], REFERENCE_SEMANTICS)
        reference = result["catalog_reference"]
        self.assertEqual(
            reference["sources"]["musicbrainz"]["source_ids"],
            ["mb-reference"],
        )
        self.assertEqual(
            reference["sources"]["apple"]["source_ids"], ["301"]
        )
        self.assertEqual(len(archive), 4)

    def test_multiple_shared_non_emitted_artists_are_ambiguous(self) -> None:
        title = "Ambiguous Song"
        artist = "Wrong Artist"
        alternatives = [
            (title, "Artist A", "mb-a"),
            (title, "Artist B", "mb-b"),
        ]
        apple_alternatives = [
            (title, "Artist A", "401"),
            (title, "Artist B", "402"),
        ]
        result, _, _ = offline_verify(
            title,
            artist,
            catalog_routes(title, artist, alternatives, apple_alternatives),
        )
        self.assertEqual(result["catalog_label"], "ambiguous")
        self.assertEqual(
            result["classification_reason"],
            "multiple_shared_non_emitted_artists",
        )
        self.assertEqual(len(result["shared_non_emitted_artists"]), 2)
        self.assertIsNone(result["reference_artist"])

    def test_no_shared_artist_is_excluded(self) -> None:
        title = "Divergent Song"
        artist = "Wrong Artist"
        result, _, _ = offline_verify(
            title,
            artist,
            catalog_routes(
                title,
                artist,
                [(title, "MusicBrainz Artist", "mb-only")],
                [(title, "Apple Artist", "501")],
            ),
        )
        self.assertEqual(result["catalog_label"], "excluded")
        self.assertEqual(
            result["classification_reason"], "no_shared_non_emitted_artist"
        )

    def test_title_missing_from_both_sources_is_excluded(self) -> None:
        title = "Missing Song"
        artist = "Missing Artist"
        result, _, _ = offline_verify(
            title,
            artist,
            catalog_routes(title, artist, [], []),
        )
        self.assertEqual(result["catalog_label"], "excluded")
        self.assertEqual(result["classification_reason"], "normalized_title_not_found")

    def test_alias_only_support_is_not_promoted_to_a_strict_label(self) -> None:
        title = "Alias Song"
        emitted = "Canonical Name"
        alias = "Stage Name"
        result, _, _ = offline_verify(
            title,
            emitted,
            catalog_routes(
                title,
                emitted,
                [(title, alias, "mb-alias")],
                [(title, alias, "601")],
            ),
            accepted_artists=[alias],
        )
        self.assertEqual(result["catalog_label"], "excluded")
        self.assertEqual(
            result["classification_reason"],
            "emitted_artist_alias_support_is_not_strict",
        )

    def test_truncated_musicbrainz_window_cannot_support_strict_conflict(self) -> None:
        title = "Window Song"
        emitted = "Wrong Artist"
        reference = "Reference Artist"
        routes = catalog_routes(
            title,
            emitted,
            [(title, reference, "mb-window")],
            [(title, reference, "apple-window")],
        )
        payload = musicbrainz_payload([(title, reference, "mb-window")])
        payload.update({"count": 101, "offset": 0})
        routes[musicbrainz_url(title, emitted)] = response(payload)
        result, _, _ = offline_verify(title, emitted, routes)
        self.assertEqual(result["catalog_label"], "excluded")
        self.assertEqual(
            result["classification_reason"], "catalog_response_window_incomplete"
        )

    def test_musicbrainz_targeted_reference_is_conservative_alias_exclusion(self) -> None:
        title = "Alias Audit Song"
        emitted = "Alternate Name"
        reference = "Canonical Artist"
        targeted = response(
            musicbrainz_payload([(title, reference, "mb-targeted-alias")])
        )
        routes = catalog_routes(
            title,
            emitted,
            [(title, reference, "mb-reference")],
            [(title, reference, "apple-reference")],
            musicbrainz_targeted=targeted,
        )
        result, _, _ = offline_verify(title, emitted, routes)
        self.assertEqual(result["catalog_label"], "excluded")
        self.assertEqual(
            result["classification_reason"],
            "musicbrainz_targeted_query_indicates_possible_alias",
        )
        self.assertTrue(result["alias_audit_complete"])

    def test_strict_conflict_records_current_complete_verifier(self) -> None:
        title = "Versioned Song"
        emitted = "Wrong Artist"
        reference = "Reference Artist"
        result, _, _ = offline_verify(
            title,
            emitted,
            catalog_routes(
                title,
                emitted,
                [(title, reference, "mb-version")],
                [(title, reference, "apple-version")],
            ),
        )
        self.assertEqual(result["catalog_label"], "strict_conflict")
        self.assertEqual(result["catalog_verifier_version"], CATALOG_VERIFIER_VERSION)
        self.assertTrue(result["catalog_evidence_complete_for_label"])
        self.assertTrue(result["alias_audit_complete"])

    def test_empty_normalized_entity_is_excluded_without_requests(self) -> None:
        fetcher = FakeFetcher({})
        archive: list[dict[str, Any]] = []
        result = verify_pair(
            "!!!",
            "Artist",
            fetcher=fetcher,
            archive=archive,
            sleep_seconds=0.0,
        )
        self.assertEqual(result["catalog_label"], "excluded")
        self.assertEqual(result["classification_reason"], "empty_normalized_title")
        self.assertEqual(fetcher.calls, [])
        self.assertEqual(archive, [])

    def test_missing_source_evidence_is_error(self) -> None:
        result = classify_phase2_catalog_evidence(
            "Song",
            "Artist",
            [
                {
                    "source": "musicbrainz",
                    "status": "ok",
                    "emitted_supported": True,
                    "title_artist_evidence": [],
                }
            ],
        )
        self.assertEqual(result["catalog_label"], "error")
        self.assertEqual(result["classification_reason"], "missing_catalog_source:apple")


class EvidenceArchiveTests(unittest.TestCase):
    def test_atomic_jsonl_write_retries_transient_windows_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            real_replace = __import__("os").replace
            attempts = 0

            def flaky_replace(source: str, target: str) -> None:
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError("transient lock")
                real_replace(source, target)

            with mock.patch("verify_phase2_catalog.os.replace", side_effect=flaky_replace):
                write_jsonl(path, [{"ok": True}])
            self.assertEqual(attempts, 3)
            self.assertEqual(read_jsonl(path), [{"ok": True}])

    def test_append_jsonl_adds_only_new_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            write_jsonl(path, [{"request_id": "first"}])
            self.assertEqual(append_jsonl(path, [{"request_id": "second"}]), 1)
            self.assertEqual(
                read_jsonl(path),
                [{"request_id": "first"}, {"request_id": "second"}],
            )

    def test_archives_complete_raw_body_url_timestamp_status_and_source_ids(self) -> None:
        title = "Evidence Song"
        artist = "Evidence Artist"
        mb_raw = json.dumps(
            musicbrainz_payload(
                [
                    (title, artist, "mb-main"),
                    ("Other Song", "Other Artist", "mb-other"),
                ]
            ),
            indent=2,
        )
        apple_raw = json.dumps(
            apple_payload([(title, artist, "701")]), indent=1
        )
        routes = {
            musicbrainz_url(title, artist): response(raw_body=mb_raw),
            apple_url(title, artist): response(raw_body=apple_raw),
        }
        result, archive, _ = offline_verify(title, artist, routes)
        self.assertEqual(result["catalog_label"], "strict_exact")
        mb_record = next(row for row in archive if row["source"] == "musicbrainz")
        required = {
            "raw_response_body",
            "request_url",
            "query_parameters",
            "queried_at_utc",
            "http_status",
            "request_status",
            "source_ids",
        }
        self.assertTrue(required.issubset(mb_record))
        self.assertEqual(mb_record["raw_response_body"], mb_raw)
        self.assertEqual(mb_record["request_url"], musicbrainz_url(title, artist))
        self.assertEqual(mb_record["queried_at_utc"], FIXED_TIME)
        self.assertEqual(mb_record["http_status"], 200)
        self.assertEqual(mb_record["request_status"], "ok")
        self.assertEqual(mb_record["source_ids"], ["mb-main", "mb-other"])
        self.assertIn("query", mb_record["query_parameters"])

    def test_http_failure_body_is_archived_and_row_is_error(self) -> None:
        title = "Failure Song"
        artist = "Artist"
        failure_body = '{"error":"temporarily unavailable"}'
        routes = catalog_routes(
            title,
            artist,
            [],
            [(title, artist, "801")],
        )
        routes[musicbrainz_url(title, artist)] = response(
            raw_body=failure_body,
            status=503,
            error="HTTPError: unavailable",
        )
        result, archive, _ = offline_verify(title, artist, routes)
        self.assertEqual(result["catalog_label"], "error")
        self.assertEqual(
            result["classification_reason"], "catalog_query_error:musicbrainz"
        )
        failure = next(row for row in archive if row["http_status"] == 503)
        self.assertEqual(failure["raw_response_body"], failure_body)
        self.assertEqual(failure["request_status"], "error")
        self.assertEqual(failure["source_ids"], [])
        self.assertEqual(failure["error_type"], "http")

    def test_targeted_query_failure_makes_entire_source_an_error(self) -> None:
        title = "Targeted Failure"
        artist = "Wrong Artist"
        targeted_url = musicbrainz_url(title, artist, targeted=True)
        routes = catalog_routes(
            title,
            artist,
            [(title, "Reference Artist", "mb-reference")],
            [(title, "Reference Artist", "901")],
            musicbrainz_targeted=response(
                raw_body='{"error":"busy"}', status=500
            ),
        )
        result, archive, _ = offline_verify(title, artist, routes)
        self.assertEqual(result["catalog_label"], "error")
        targeted = next(row for row in archive if row["request_url"] == targeted_url)
        self.assertEqual(targeted["query_kind"], "title_artist")
        self.assertEqual(targeted["http_status"], 500)
        mb_source = next(
            row for row in result["catalog_sources"] if row["source"] == "musicbrainz"
        )
        self.assertEqual(mb_source["status"], "error")
        self.assertIn("mb-reference", mb_source["source_ids"])

    def test_invalid_json_keeps_raw_body_and_is_error(self) -> None:
        title = "Malformed Response"
        artist = "Artist"
        routes = catalog_routes(
            title,
            artist,
            [],
            [(title, artist, "1001")],
        )
        routes[musicbrainz_url(title, artist)] = response(raw_body="not-json")
        result, archive, _ = offline_verify(title, artist, routes)
        self.assertEqual(result["catalog_label"], "error")
        malformed = next(row for row in archive if row["source"] == "musicbrainz")
        self.assertEqual(malformed["raw_response_body"], "not-json")
        self.assertEqual(malformed["http_status"], 200)
        self.assertEqual(malformed["request_status"], "error")
        self.assertEqual(malformed["error_type"], "response_parse")

    def test_each_retry_attempt_is_archived(self) -> None:
        title = "Retry Song"
        artist = "Retry Artist"
        mb_url = musicbrainz_url(title, artist)
        routes = {
            mb_url: [
                response(raw_body='{"error":"busy"}', status=503),
                response(musicbrainz_payload([(title, artist, "mb-retry")])),
            ],
            apple_url(title, artist): response(
                apple_payload([(title, artist, "1101")])
            ),
        }
        result, archive, fetcher = offline_verify(
            title, artist, routes, max_attempts=2
        )
        self.assertEqual(result["catalog_label"], "strict_exact")
        mb_records = [row for row in archive if row["source"] == "musicbrainz"]
        self.assertEqual([row["attempt"] for row in mb_records], [1, 2])
        self.assertEqual(
            [row["request_status"] for row in mb_records], ["error", "ok"]
        )
        self.assertEqual(fetcher.calls.count(mb_url), 2)

    def test_successful_responses_are_reused_from_injected_cache(self) -> None:
        title = "Cached Song"
        artist = "Cached Artist"
        routes = catalog_routes(
            title,
            artist,
            [(title, artist, "mb-cache")],
            [(title, artist, "1201")],
        )
        fetcher = FakeFetcher(routes)
        archive: list[dict[str, Any]] = []
        cache: dict[str, dict[str, Any]] = {}
        session = CatalogRequestSession(
            fetcher=fetcher,
            archive=archive,
            cache=cache,
            now=lambda: FIXED_TIME,
            sleep_seconds=0.0,
            max_attempts=1,
        )
        first = verify_pair(title, artist, session=session)
        second = verify_pair(title, artist, session=session)
        self.assertEqual(first["catalog_label"], "strict_exact")
        self.assertEqual(second["catalog_label"], "strict_exact")
        self.assertEqual(len(fetcher.calls), 2)
        self.assertEqual(len(archive), 2)
        self.assertEqual(len(cache), 2)
        self.assertTrue(
            all(
                source["cache_hit_count"] == 1
                for source in second["catalog_sources"]
            )
        )


class ClusterAndCliTests(unittest.TestCase):
    def test_cluster_summary_aggregates_without_selecting_a_row(self) -> None:
        rows = [
            {
                "record_id": "a",
                "title": "Cluster Song",
                "artist": "Artist A",
                "normalized_title": "clustersong",
                "normalized_artist": "artista",
                "phase2_catalog_label": "strict_exact",
            },
            {
                "record_id": "b",
                "title": "Cluster-Song",
                "artist": "Wrong Artist",
                "normalized_title": "clustersong",
                "normalized_artist": "wrongartist",
                "phase2_catalog_label": "strict_conflict",
                "catalog_reference": {
                    "artist": "Artist A",
                    "normalized_artist": "artista",
                    "semantics": REFERENCE_SEMANTICS,
                    "sources": {},
                },
            },
            {
                "record_id": "c",
                "title": "Other Song",
                "artist": "Artist C",
                "normalized_title": "othersong",
                "normalized_artist": "artistc",
                "phase2_catalog_label": "excluded",
            },
        ]
        apply_title_cluster_summaries(rows)
        cluster = rows[0]["catalog_title_cluster"]
        self.assertEqual(cluster, rows[1]["catalog_title_cluster"])
        self.assertEqual(cluster["row_count"], 2)
        self.assertEqual(
            cluster["catalog_label_counts"],
            {"strict_conflict": 1, "strict_exact": 1},
        )
        self.assertEqual(cluster["confirmatory_eligible_row_count"], 2)
        self.assertEqual(len(cluster["strict_conflict_references"]), 1)
        self.assertNotIn("selected", json.dumps(rows, sort_keys=True))

    def test_cli_writes_rows_evidence_cache_and_resumes_without_network(self) -> None:
        title = "CLI Song"
        artist = "CLI Artist"
        routes = catalog_routes(
            title,
            artist,
            [(title, artist, "mb-cli")],
            [(title, artist, "1301")],
        )
        fetcher = FakeFetcher(routes)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            evidence_path = root / "evidence.jsonl"
            cache_path = root / "cache.jsonl"
            write_jsonl(
                input_path,
                [
                    {
                        "record_type": "generated_pair",
                        "record_id": "row-1",
                        "title": title,
                        "artist": artist,
                    },
                    {
                        "record_type": "generated_pair",
                        "record_id": "row-2",
                        "title": title,
                        "artist": artist,
                    },
                ],
            )
            argv = [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--evidence-archive",
                str(evidence_path),
                "--cache",
                str(cache_path),
                "--sleep-seconds",
                "0",
                "--max-attempts",
                "1",
            ]
            self.assertEqual(
                run_cli(argv, fetcher=fetcher, now=lambda: FIXED_TIME), 0
            )
            output_rows = read_jsonl(output_path)
            evidence_rows = read_jsonl(evidence_path)
            cache_rows = read_jsonl(cache_path)
            self.assertEqual(
                [row["catalog_label"] for row in output_rows],
                ["strict_exact", "strict_exact"],
            )
            self.assertEqual(
                output_rows[0]["catalog_title_cluster"]["row_count"], 2
            )
            self.assertEqual(len(evidence_rows), 2)
            self.assertEqual(len(cache_rows), 2)
            self.assertEqual(len(fetcher.calls), 2)
            self.assertNotIn(
                "raw_response_body", json.dumps(output_rows, sort_keys=True)
            )
            self.assertTrue(
                all("raw_response_body" in row for row in evidence_rows)
            )

            def no_network(_: str) -> Any:
                raise AssertionError("resume attempted a network request")

            self.assertEqual(
                run_cli(argv, fetcher=no_network, now=lambda: FIXED_TIME), 0
            )
            self.assertEqual(read_jsonl(output_path), output_rows)
            self.assertEqual(read_jsonl(evidence_path), evidence_rows)

    def test_first_row_interruption_reuses_evidence_without_output_checkpoint(self) -> None:
        title = "Recovered First Song"
        artist = "Recovered Artist"
        routes = catalog_routes(
            title,
            artist,
            [(title, artist, "mb-recovered")],
            [(title, artist, "apple-recovered")],
        )
        _, archived, _ = offline_verify(title, artist, routes)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            evidence_path = root / "evidence.jsonl"
            cache_path = root / "cache.jsonl"
            write_jsonl(
                input_path,
                [{"record_id": "first", "title": title, "artist": artist}],
            )
            write_jsonl(evidence_path, archived)
            stale_cache = []
            for index, row in enumerate(archived):
                empty_payload = (
                    musicbrainz_payload([])
                    if row["source"] == "musicbrainz"
                    else apple_payload([])
                )
                stale_cache.append(
                    {
                        **row,
                        "request_id": f"stale-{index}",
                        "raw_response_body": json.dumps(empty_payload),
                        "source_ids": [],
                    }
                )
            write_jsonl(cache_path, stale_cache)

            def no_network(_: str) -> Any:
                raise AssertionError("archived first-row evidence was not reused")

            argv = [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--evidence-archive",
                str(evidence_path),
                "--cache",
                str(cache_path),
                "--sleep-seconds",
                "0",
                "--max-attempts",
                "1",
            ]
            self.assertEqual(run_cli(argv, fetcher=no_network), 0)
            self.assertEqual(read_jsonl(output_path)[0]["catalog_label"], "strict_exact")
            self.assertEqual(read_jsonl(evidence_path), archived)

    def test_resume_rejects_changed_relation_even_with_same_record_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            evidence_path = root / "evidence.jsonl"
            write_jsonl(
                input_path,
                [{"record_id": "same-id", "title": "Song", "artist": "Artist A"}],
            )
            write_jsonl(
                output_path,
                [{"record_id": "same-id", "title": "Song", "artist": "Artist B"}],
            )
            write_jsonl(evidence_path, [])
            with self.assertRaisesRegex(ValueError, "checkpoint row 0"):
                run_cli(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--evidence-archive",
                        str(evidence_path),
                    ],
                    fetcher=lambda _: response({}),
                )

    def test_source_files_are_ascii(self) -> None:
        paths = [
            SCRIPTS / "verify_phase2_catalog.py",
            Path(__file__),
        ]
        for path in paths:
            path.read_bytes().decode("ascii")


if __name__ == "__main__":
    unittest.main()
