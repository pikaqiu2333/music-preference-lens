"""Verify generated title-artist pairs against MusicBrainz and Apple Search."""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "MusicPreferenceLens/0.1 (research contact: research-contact@example.invalid)"


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if character.isalnum())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def request_json(url: str, timeout: float = 20.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
            time.sleep(2**attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("catalog request retries exhausted")


def musicbrainz_url(title: str, artist: str, *, targeted: bool = False) -> str:
    query = f'recording:"{title}"'
    if targeted:
        query += f' AND artist:"{artist}"'
    return "https://musicbrainz.org/ws/2/recording/?" + urllib.parse.urlencode(
        {"query": query, "fmt": "json", "limit": 100}
    )


def apple_url(title: str, artist: str, *, targeted: bool = False) -> str:
    term = f"{title} {artist}" if targeted else title
    return "https://itunes.apple.com/search?" + urllib.parse.urlencode(
        {"term": term, "entity": "song", "limit": 200}
    )


def musicbrainz_candidates(payload: dict[str, Any]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for recording in payload.get("recordings", []):
        artists = []
        for credit in recording.get("artist-credit", []):
            if isinstance(credit, dict):
                artists.append(credit.get("name") or credit.get("artist", {}).get("name", ""))
        candidates.append(
            {
                "title": recording.get("title", ""),
                "artist": " ".join(part for part in artists if part),
                "source_id": recording.get("id", ""),
            }
        )
    return candidates


def apple_candidates(payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "title": result.get("trackName", ""),
            "artist": result.get("artistName", ""),
            "source_id": str(result.get("trackId", "")),
        }
        for result in payload.get("results", [])
        if result.get("wrapperType") == "track" or result.get("kind") == "song"
    ]


def candidate_evidence(
    title: str,
    artist: str,
    candidates: list[dict[str, str]],
    accepted_artists: list[str] | None = None,
) -> dict[str, Any]:
    title_key = normalize_name(title)
    artist_keys = {
        normalize_name(value) for value in [artist, *(accepted_artists or [])] if value
    }
    exact = []
    title_matches = []
    for candidate in candidates:
        candidate_title = normalize_name(candidate.get("title", ""))
        candidate_artist = normalize_name(candidate.get("artist", ""))
        if candidate_title == title_key:
            title_matches.append(candidate)
            if candidate_artist in artist_keys:
                exact.append(candidate)
    return {
        "exact": exact[:5],
        "title_matches": title_matches[:10],
    }


def classify_catalog_evidence(source_rows: list[dict[str, Any]]) -> str:
    successful = [row for row in source_rows if row["status"] == "ok"]
    if any(row["evidence"]["exact"] for row in successful):
        return "verified_exact"
    if len(successful) != len(source_rows):
        return "verification_error"
    if any(row["evidence"]["title_matches"] for row in successful):
        return "catalog_conflict"
    return "unverified"


def verify_pair(
    title: str,
    artist: str,
    *,
    accepted_artists: list[str] | None = None,
    fetcher: Callable[[str], dict[str, Any]] = request_json,
    sleep_seconds: float = 1.1,
) -> dict[str, Any]:
    sources = [
        ("musicbrainz", musicbrainz_url, musicbrainz_candidates),
        ("apple", apple_url, apple_candidates),
    ]
    source_rows: list[dict[str, Any]] = []
    for source, url_builder, parser in sources:
        try:
            payload = fetcher(url_builder(title, artist))
            candidates = parser(payload)
            evidence = candidate_evidence(
                title,
                artist,
                candidates,
                accepted_artists=accepted_artists,
            )
            if not evidence["exact"]:
                if source == "musicbrainz" and sleep_seconds:
                    time.sleep(sleep_seconds)
                targeted_payload = fetcher(url_builder(title, artist, targeted=True))
                candidates.extend(parser(targeted_payload))
                evidence = candidate_evidence(
                    title,
                    artist,
                    candidates,
                    accepted_artists=accepted_artists,
                )
            source_rows.append(
                {
                    "source": source,
                    "status": "ok",
                    "evidence": evidence,
                }
            )
        except Exception as exc:  # Network failures must remain explicit evidence.
            source_rows.append(
                {
                    "source": source,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "evidence": {"exact": [], "title_matches": []},
                }
            )
        if source == "musicbrainz" and sleep_seconds:
            time.sleep(sleep_seconds)
    return {
        "catalog_label": classify_catalog_evidence(source_rows),
        "catalog_sources": source_rows,
    }


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    generated = [row for row in rows if row.get("record_type") == "generated_pair"]
    reported = generated or [row for row in rows if row.get("catalog_label")]
    counts = Counter(row.get("catalog_label", "missing") for row in reported)
    control_rows = [row for row in reported if row.get("group")]
    for row in control_rows:
        expected_label = {
            "known_exact": "verified_exact",
            "artist_mismatch": "catalog_conflict",
        }.get(row["group"])
        row["control_expectation_ok"] = (
            row["catalog_label"] == expected_label
            if expected_label
            else row["catalog_label"] in {"unverified", "catalog_conflict"}
        )
    lines = [
        "# Song Entity Catalog Verification",
        "",
        f"- Reported pairs: {len(reported)}",
        f"- Verified exact: {counts['verified_exact']}",
        f"- Catalog conflicts: {counts['catalog_conflict']}",
        f"- Unverified: {counts['unverified']}",
        f"- Verification errors: {counts['verification_error']}",
        "",
        "A catalog miss is reported as `unverified`, not as proof that a pair is",
        "fictional.",
        "",
        "| Context/group | Title | Artist | Catalog label | Generation knownness | Neutral Unknown logit | Self-attributed Unknown logit |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    if control_rows:
        lines.insert(7, f"- Control expectations passed: {sum(row['control_expectation_ok'] for row in control_rows)}/{len(control_rows)}")
    for row in reported:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("context_id") or row.get("group", "")),
                    str(row.get("title", "")).replace("|", "\\|"),
                    str(row.get("artist", "")).replace("|", "\\|"),
                    str(row.get("catalog_label", "")),
                    f"{float(row.get('generation_knownness', 0.0)):.3f}",
                    f"{float(row.get('neutral_unknown_logit', 0.0)):.3f}",
                    f"{float(row.get('self_attributed_unknown_logit', 0.0)):.3f}",
                ]
            )
            + " |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "song_entity_catalog_verified.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "song_entity_catalog_verification.md",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="skip this many selected pairs before applying --limit",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="verify every row with title/artist fields instead of generated rows only",
    )
    parser.add_argument(
        "--retry-errors-only",
        action="store_true",
        help="re-query only rows already labeled verification_error",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    response_cache: dict[str, dict[str, Any]] = {}
    last_musicbrainz_request = 0.0

    def cached_fetcher(url: str) -> dict[str, Any]:
        nonlocal last_musicbrainz_request
        if url in response_cache:
            return response_cache[url]
        if "musicbrainz.org" in url:
            elapsed = time.monotonic() - last_musicbrainz_request
            if elapsed < 1.1:
                time.sleep(1.1 - elapsed)
            last_musicbrainz_request = time.monotonic()
        payload = request_json(url)
        response_cache[url] = payload
        return payload

    generated_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("record_type") == "generated_pair"
        or (args.all_pairs and row.get("title") and row.get("artist"))
    ]
    if args.retry_errors_only:
        generated_indices = [
            index
            for index in generated_indices
            if rows[index].get("catalog_label") == "verification_error"
        ]
    if args.offset:
        generated_indices = generated_indices[args.offset :]
    if args.limit is not None:
        generated_indices = generated_indices[: args.limit]

    def checkpoint() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for index in generated_indices:
        rows[index].update(
            verify_pair(
                rows[index]["title"],
                rows[index]["artist"],
                accepted_artists=rows[index].get("accepted_artists"),
                fetcher=cached_fetcher,
                sleep_seconds=0.0,
            )
        )
        checkpoint()

    checkpoint()
    write_report(args.report, rows)
    print(f"Verified {len(generated_indices)} pairs")
    print(f"- {args.output}")
    print(f"- {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
