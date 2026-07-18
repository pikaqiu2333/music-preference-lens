"""Verify Phase 2 title-artist relations against two independent catalogs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from verify_song_entity_catalog import (
    USER_AGENT,
    apple_candidates,
    apple_url,
    musicbrainz_candidates,
    musicbrainz_url,
    normalize_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = ("musicbrainz", "apple")
STRICT_LABELS = {"strict_exact", "strict_conflict"}
FINAL_LABELS = STRICT_LABELS | {"ambiguous", "excluded", "error"}
TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}
REFERENCE_SEMANTICS = "catalog-supported reference, not unique real-world answer"
CATALOG_VERIFIER_VERSION = "phase2_catalog_v2_complete_alias_audit"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        for attempt in range(8):
            try:
                os.replace(temporary_name, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def request_raw(url: str, timeout: float = 20.0) -> dict[str, Any]:
    """Perform one HTTP attempt and retain the response body without parsing it."""

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            return {
                "http_status": int(status),
                "raw_response_body": response.read().decode(
                    "utf-8", errors="replace"
                ),
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        return {
            "http_status": int(exc.code),
            "raw_response_body": exc.read().decode("utf-8", errors="replace"),
            "error": f"HTTPError: {exc}",
        }


def _json_body(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _body_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return _json_body(value)


def _normalize_fetch_result(value: Any) -> tuple[int | None, str, str | None]:
    if isinstance(value, Mapping) and (
        "raw_response_body" in value or "raw_body" in value
    ):
        raw_body = value.get("raw_response_body", value.get("raw_body", ""))
        status_value = value.get("http_status", value.get("status_code", 200))
        try:
            status = int(status_value) if status_value is not None else None
        except (TypeError, ValueError):
            status = None
        error = value.get("error")
        return status, _body_text(raw_body), str(error) if error else None
    if isinstance(value, tuple) and len(value) == 2:
        status_value, raw_body = value
        try:
            status = int(status_value) if status_value is not None else None
        except (TypeError, ValueError):
            status = None
        return status, _body_text(raw_body), None
    if isinstance(value, bytes):
        return 200, value.decode("utf-8", errors="replace"), None
    if isinstance(value, str):
        return 200, value, None
    return 200, _json_body(value), None


def _exception_response(exc: BaseException) -> tuple[int | None, str]:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return int(exc.code), body
    return None, ""


def _parse_payload(
    raw_body: str,
    parser: Callable[[dict[str, Any]], list[dict[str, str]]],
) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    payload = json.loads(raw_body)
    if not isinstance(payload, dict):
        raise ValueError("catalog response must be a JSON object")
    candidates = parser(payload)
    source_ids = sorted(
        {
            str(candidate.get("source_id", ""))
            for candidate in candidates
            if candidate.get("source_id", "")
        }
    )
    return payload, candidates, source_ids


def _query_parameters(url: str) -> dict[str, list[str]]:
    return {
        key: values
        for key, values in urllib.parse.parse_qs(
            urllib.parse.urlsplit(url).query, keep_blank_values=True
        ).items()
    }


def _first_integer(
    value: Any, *, default: int | None = None
) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def response_completeness(
    source: str, payload: Mapping[str, Any], url: str
) -> dict[str, Any]:
    """State whether one catalog response proves a complete result window."""

    parameters = _query_parameters(url)
    limit = _first_integer((parameters.get("limit") or [None])[0])
    if source == "musicbrainz":
        returned = len(payload.get("recordings", []))
        offset = _first_integer(
            payload.get("offset"),
            default=_first_integer((parameters.get("offset") or [0])[0], default=0),
        )
        total = _first_integer(payload.get("count"))
        complete = (
            bool(total is not None and offset is not None and offset + returned >= total)
            if total is not None
            else bool(limit is None or returned < limit)
        )
    elif source == "apple":
        returned = len(payload.get("results", []))
        offset = 0
        total = _first_integer(payload.get("resultCount"), default=returned)
        # Apple Search has no stable pagination contract. A full limit-sized page is
        # therefore treated as truncated rather than as negative evidence.
        complete = bool(limit is None or max(returned, total or 0) < limit)
    else:
        raise ValueError(f"unsupported catalog source: {source}")
    return {
        "complete": complete,
        "returned_result_count": returned,
        "reported_total_count": total,
        "offset": offset,
        "limit": limit,
    }


class CatalogRequestSession:
    """Archive, retry, throttle, and cache catalog HTTP attempts."""

    def __init__(
        self,
        *,
        fetcher: Callable[[str], Any] = request_raw,
        cache: dict[str, dict[str, Any]] | None = None,
        archive: list[dict[str, Any]] | None = None,
        now: Callable[[], str] = utc_timestamp,
        sleep_seconds: float = 1.1,
        max_attempts: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
        on_record: Callable[[], None] | None = None,
    ) -> None:
        if sleep_seconds < 0:
            raise ValueError("sleep_seconds must be non-negative")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.fetcher = fetcher
        self.cache = cache if cache is not None else {}
        self.archive = archive if archive is not None else []
        self.now = now
        self.sleep_seconds = sleep_seconds
        self.max_attempts = max_attempts
        self.sleep_fn = sleep_fn
        self.on_record = on_record
        self._archive_ids = {
            str(row.get("request_id", ""))
            for row in self.archive
            if row.get("request_id")
        }
        self._last_musicbrainz_request = 0.0

    def _remember(self, record: dict[str, Any]) -> None:
        request_id = str(record["request_id"])
        if request_id not in self._archive_ids:
            self.archive.append(record)
            self._archive_ids.add(request_id)
            if self.on_record is not None:
                self.on_record()

    def _wait(self, source: str) -> None:
        if source != "musicbrainz" or not self.sleep_seconds:
            return
        elapsed = time.monotonic() - self._last_musicbrainz_request
        if self._last_musicbrainz_request and elapsed < self.sleep_seconds:
            self.sleep_fn(self.sleep_seconds - elapsed)
        self._last_musicbrainz_request = time.monotonic()

    @staticmethod
    def _retryable(record: Mapping[str, Any]) -> bool:
        status = record.get("http_status")
        return status is None or status in TRANSIENT_HTTP_STATUSES

    def _cached_result(
        self,
        source: str,
        url: str,
        parser: Callable[[dict[str, Any]], list[dict[str, str]]],
    ) -> dict[str, Any] | None:
        record = self.cache.get(url)
        if not record or record.get("request_status") != "ok":
            return None
        try:
            payload, candidates, source_ids = _parse_payload(
                str(record["raw_response_body"]), parser
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.cache.pop(url, None)
            return None
        if record.get("source") != source:
            self.cache.pop(url, None)
            return None
        self._remember(record)
        return {
            "ok": True,
            "payload": payload,
            "candidates": candidates,
            "source_ids": source_ids,
            "records": [record],
            "cache_hit": True,
        }

    def request(
        self,
        *,
        source: str,
        query_kind: str,
        url: str,
        parser: Callable[[dict[str, Any]], list[dict[str, str]]],
    ) -> dict[str, Any]:
        cached = self._cached_result(source, url, parser)
        if cached is not None:
            return cached

        attempt_records: list[dict[str, Any]] = []
        for attempt in range(1, self.max_attempts + 1):
            self._wait(source)
            queried_at = self.now()
            http_status: int | None = None
            raw_body = ""
            supplied_error: str | None = None
            error: str | None = None
            error_type: str | None = None
            payload: dict[str, Any] | None = None
            candidates: list[dict[str, str]] = []
            source_ids: list[str] = []
            try:
                fetched = self.fetcher(url)
                http_status, raw_body, supplied_error = _normalize_fetch_result(
                    fetched
                )
                if http_status is None or not 200 <= http_status < 300:
                    error = supplied_error or f"HTTP status {http_status}"
                    error_type = "http"
                else:
                    try:
                        payload, candidates, source_ids = _parse_payload(
                            raw_body, parser
                        )
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        error_type = "response_parse"
            except Exception as exc:
                http_status, raw_body = _exception_response(exc)
                error = f"{type(exc).__name__}: {exc}"
                error_type = "network"

            request_status = "ok" if error is None else "error"
            record = {
                "record_type": "phase2_catalog_request",
                "request_id": uuid.uuid4().hex,
                "source": source,
                "query_kind": query_kind,
                "request_url": url,
                "query_parameters": _query_parameters(url),
                "queried_at_utc": queried_at,
                "attempt": attempt,
                "http_status": http_status,
                "request_status": request_status,
                "raw_response_body": raw_body,
                "source_ids": source_ids,
                "error_type": error_type,
                "error": error,
            }
            if request_status == "ok":
                self.cache[url] = record
            self._remember(record)
            attempt_records.append(record)
            if request_status == "ok":
                return {
                    "ok": True,
                    "payload": payload,
                    "candidates": candidates,
                    "source_ids": source_ids,
                    "records": attempt_records,
                    "cache_hit": False,
                }
            if attempt == self.max_attempts or not self._retryable(record):
                break
            self.sleep_fn(float(2 ** (attempt - 1)))

        return {
            "ok": False,
            "payload": None,
            "candidates": [],
            "source_ids": [],
            "records": attempt_records,
            "cache_hit": False,
        }


def _dedupe_candidates(
    candidates: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        item = {
            "title": str(candidate.get("title", "")),
            "artist": str(candidate.get("artist", "")),
            "source_id": str(candidate.get("source_id", "")),
        }
        key = (
            normalize_name(item["title"]),
            normalize_name(item["artist"]),
            item["source_id"],
        )
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def _artist_index(
    title: str, candidates: Sequence[dict[str, str]]
) -> dict[str, dict[str, list[str]]]:
    title_key = normalize_name(title)
    index: dict[str, dict[str, list[str]]] = {}
    for candidate in candidates:
        if normalize_name(candidate.get("title", "")) != title_key:
            continue
        artist = str(candidate.get("artist", ""))
        artist_key = normalize_name(artist)
        if not artist_key:
            continue
        entry = index.setdefault(
            artist_key, {"artist_names": [], "source_ids": []}
        )
        if artist and artist not in entry["artist_names"]:
            entry["artist_names"].append(artist)
        source_id = str(candidate.get("source_id", ""))
        if source_id and source_id not in entry["source_ids"]:
            entry["source_ids"].append(source_id)
    return index


def _public_artist_index(
    index: Mapping[str, Mapping[str, Sequence[str]]]
) -> list[dict[str, Any]]:
    return [
        {
            "normalized_artist": artist_key,
            "artist_names": list(index[artist_key]["artist_names"]),
            "source_ids": list(index[artist_key]["source_ids"]),
        }
        for artist_key in sorted(index)
    ]


def _index_from_source(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["normalized_artist"]): dict(item)
        for item in row.get("title_artist_evidence", [])
    }


def _source_error(records: Sequence[Mapping[str, Any]]) -> str:
    errors = [str(record.get("error")) for record in records if record.get("error")]
    return "; ".join(errors) if errors else "catalog query failed"


def verify_source(
    source: str,
    title: str,
    artist: str,
    *,
    session: CatalogRequestSession,
    accepted_artists: Sequence[str] | None = None,
) -> dict[str, Any]:
    if source == "musicbrainz":
        url_builder = musicbrainz_url
        parser = musicbrainz_candidates
    elif source == "apple":
        url_builder = apple_url
        parser = apple_candidates
    else:
        raise ValueError(f"unsupported catalog source: {source}")

    candidates: list[dict[str, str]] = []
    targeted_candidates: list[dict[str, str]] = []
    request_records: list[dict[str, Any]] = []
    query_windows: list[dict[str, Any]] = []
    cache_hit_count = 0
    broad_url = url_builder(title, artist)
    broad = session.request(
        source=source,
        query_kind="title",
        url=broad_url,
        parser=parser,
    )
    request_records.extend(broad["records"])
    cache_hit_count += int(broad["cache_hit"])
    failed = not broad["ok"]
    if broad["ok"]:
        candidates.extend(broad["candidates"])
        query_windows.append(
            {
                "query_kind": "title",
                **response_completeness(source, broad["payload"], broad_url),
            }
        )

    emitted_key = normalize_name(artist)
    if not failed and emitted_key not in _artist_index(title, candidates):
        targeted_url = url_builder(title, artist, targeted=True)
        targeted = session.request(
            source=source,
            query_kind="title_artist",
            url=targeted_url,
            parser=parser,
        )
        request_records.extend(targeted["records"])
        cache_hit_count += int(targeted["cache_hit"])
        if targeted["ok"]:
            targeted_candidates = list(targeted["candidates"])
            candidates.extend(targeted_candidates)
            query_windows.append(
                {
                    "query_kind": "title_artist",
                    **response_completeness(
                        source, targeted["payload"], targeted_url
                    ),
                }
            )
        else:
            failed = True

    candidates = _dedupe_candidates(candidates)
    index = _artist_index(title, candidates)
    targeted_index = _artist_index(title, targeted_candidates)
    alias_keys = {
        normalize_name(value)
        for value in accepted_artists or []
        if value and normalize_name(value) != emitted_key
    }
    title_matches = [
        candidate
        for candidate in candidates
        if normalize_name(candidate.get("title", "")) == normalize_name(title)
    ]
    request_source_ids = sorted(
        {
            str(source_id)
            for record in request_records
            for source_id in record.get("source_ids", [])
            if source_id
        }
    )
    return {
        "source": source,
        "status": "error" if failed else "ok",
        "error": _source_error(request_records) if failed else None,
        "request_ids": [record["request_id"] for record in request_records],
        "request_urls": list(
            dict.fromkeys(record["request_url"] for record in request_records)
        ),
        "request_count": len(request_records),
        "cache_hit_count": cache_hit_count,
        "source_ids": request_source_ids,
        "title_match_count": len(title_matches),
        "title_matches": title_matches,
        "title_artist_evidence": _public_artist_index(index),
        "targeted_title_artist_evidence": _public_artist_index(targeted_index),
        "query_windows": query_windows,
        "evidence_complete_for_conflict": bool(
            not failed
            and query_windows
            and all(window["complete"] for window in query_windows)
        ),
        "emitted_supported": emitted_key in index,
        "emitted_match_source_ids": list(
            index.get(emitted_key, {}).get("source_ids", [])
        ),
        "alias_supported": any(alias_key in index for alias_key in alias_keys),
    }


def _common_artist(
    artist_key: str, source_rows: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    source_evidence: dict[str, Any] = {}
    display_artist = ""
    for source in SOURCE_NAMES:
        item = _index_from_source(source_rows[source])[artist_key]
        names = list(item.get("artist_names", []))
        ids = list(item.get("source_ids", []))
        if not display_artist and names:
            display_artist = str(names[0])
        source_evidence[source] = {
            "artist_names": names,
            "source_ids": ids,
        }
    return {
        "artist": display_artist,
        "normalized_artist": artist_key,
        "sources": source_evidence,
    }


def _base_classification(label: str, reason: str) -> dict[str, Any]:
    return {
        "catalog_verifier_version": CATALOG_VERIFIER_VERSION,
        "catalog_label": label,
        "phase2_catalog_label": label,
        "classification_reason": reason,
        "confirmatory_catalog_eligible": label in STRICT_LABELS,
        "catalog_evidence_complete_for_label": False,
        "alias_audit_complete": False,
        "reference_artist": None,
        "normalized_reference_artist": None,
        "reference_semantics": None,
        "catalog_reference": None,
        "shared_non_emitted_artists": [],
    }


def classify_phase2_catalog_evidence(
    title: str,
    artist: str,
    source_rows: Sequence[dict[str, Any]],
    *,
    accepted_artists: Sequence[str] | None = None,
) -> dict[str, Any]:
    del title
    by_source = {str(row.get("source")): row for row in source_rows}
    missing = [source for source in SOURCE_NAMES if source not in by_source]
    if missing:
        return _base_classification(
            "error", "missing_catalog_source:" + ",".join(missing)
        )
    failed = [
        source for source in SOURCE_NAMES if by_source[source].get("status") != "ok"
    ]
    if failed:
        return _base_classification(
            "error", "catalog_query_error:" + ",".join(failed)
        )

    emitted_key = normalize_name(artist)
    emitted_support = [
        bool(by_source[source].get("emitted_supported"))
        for source in SOURCE_NAMES
    ]
    if all(emitted_support):
        result = _base_classification(
            "strict_exact", "emitted_artist_supported_by_both"
        )
        result["catalog_evidence_complete_for_label"] = True
        result["alias_audit_complete"] = True
        return result
    if any(emitted_support):
        return _base_classification(
            "excluded", "emitted_artist_supported_by_one_source_only"
        )

    alias_keys = {
        normalize_name(value)
        for value in accepted_artists or []
        if value and normalize_name(value) != emitted_key
    }
    indexes = {
        source: _index_from_source(by_source[source]) for source in SOURCE_NAMES
    }
    common_keys = set(indexes["musicbrainz"]) & set(indexes["apple"])
    common_keys.discard(emitted_key)
    shared_aliases = common_keys & alias_keys
    if shared_aliases or any(
        bool(by_source[source].get("alias_supported")) for source in SOURCE_NAMES
    ):
        return _base_classification(
            "excluded", "emitted_artist_alias_support_is_not_strict"
        )

    if not all(
        bool(by_source[source].get("evidence_complete_for_conflict"))
        for source in SOURCE_NAMES
    ):
        return _base_classification(
            "excluded", "catalog_response_window_incomplete"
        )

    musicbrainz_targeted_keys = {
        str(item.get("normalized_artist", ""))
        for item in by_source["musicbrainz"].get(
            "targeted_title_artist_evidence", []
        )
        if item.get("normalized_artist")
    }
    targeted_shared_aliases = common_keys & musicbrainz_targeted_keys
    if targeted_shared_aliases:
        result = _base_classification(
            "excluded", "musicbrainz_targeted_query_indicates_possible_alias"
        )
        result["catalog_evidence_complete_for_label"] = True
        result["alias_audit_complete"] = True
        return result

    eligible_keys = sorted(common_keys - alias_keys)
    shared = [_common_artist(key, by_source) for key in eligible_keys]
    if len(shared) == 1:
        result = _base_classification(
            "strict_conflict", "one_shared_non_emitted_artist"
        )
        reference = dict(shared[0])
        reference["semantics"] = REFERENCE_SEMANTICS
        result.update(
            {
                "catalog_evidence_complete_for_label": True,
                "alias_audit_complete": True,
                "reference_artist": reference["artist"],
                "normalized_reference_artist": reference["normalized_artist"],
                "reference_semantics": REFERENCE_SEMANTICS,
                "catalog_reference": reference,
                "shared_non_emitted_artists": shared,
            }
        )
        return result
    if len(shared) > 1:
        result = _base_classification(
            "ambiguous", "multiple_shared_non_emitted_artists"
        )
        result["shared_non_emitted_artists"] = shared
        return result

    if any(by_source[source].get("title_match_count", 0) for source in SOURCE_NAMES):
        return _base_classification(
            "excluded", "no_shared_non_emitted_artist"
        )
    return _base_classification("excluded", "normalized_title_not_found")


def verify_pair(
    title: str,
    artist: str,
    *,
    accepted_artists: Sequence[str] | None = None,
    session: CatalogRequestSession | None = None,
    fetcher: Callable[[str], Any] = request_raw,
    cache: dict[str, dict[str, Any]] | None = None,
    archive: list[dict[str, Any]] | None = None,
    now: Callable[[], str] = utc_timestamp,
    sleep_seconds: float = 1.1,
    max_attempts: int = 3,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    title = str(title or "")
    artist = str(artist or "")
    title_key = normalize_name(title)
    artist_key = normalize_name(artist)
    if not title_key or not artist_key:
        reason = "empty_normalized_title" if not title_key else "empty_normalized_artist"
        result = _base_classification("excluded", reason)
        result.update(
            {
                "normalized_title": title_key,
                "normalized_artist": artist_key,
                "catalog_sources": [],
                "catalog_request_ids": [],
            }
        )
        return result

    if session is None:
        session = CatalogRequestSession(
            fetcher=fetcher,
            cache=cache,
            archive=archive,
            now=now,
            sleep_seconds=sleep_seconds,
            max_attempts=max_attempts,
            sleep_fn=sleep_fn,
        )
    source_rows = [
        verify_source(
            source,
            title,
            artist,
            session=session,
            accepted_artists=accepted_artists,
        )
        for source in SOURCE_NAMES
    ]
    result = classify_phase2_catalog_evidence(
        title,
        artist,
        source_rows,
        accepted_artists=accepted_artists,
    )
    result.update(
        {
            "normalized_title": title_key,
            "normalized_artist": artist_key,
            "catalog_sources": source_rows,
            "catalog_request_ids": [
                request_id
                for source_row in source_rows
                for request_id in source_row["request_ids"]
            ],
        }
    )
    return result


def summarize_title_clusters(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    clusters: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        title_key = str(row.get("normalized_title") or normalize_name(str(row.get("title", ""))))
        clusters.setdefault(title_key, []).append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for title_key, cluster_rows in clusters.items():
        labels = [
            str(row["phase2_catalog_label"])
            for row in cluster_rows
            if row.get("phase2_catalog_label") in FINAL_LABELS
        ]
        references: dict[str, dict[str, Any]] = {}
        for row in cluster_rows:
            reference = row.get("catalog_reference")
            if isinstance(reference, Mapping) and reference.get("normalized_artist"):
                references[str(reference["normalized_artist"])] = dict(reference)
        summaries[title_key] = {
            "normalized_title": title_key,
            "row_count": len(cluster_rows),
            "classified_row_count": len(labels),
            "pending_row_count": len(cluster_rows) - len(labels),
            "catalog_label_counts": dict(sorted(Counter(labels).items())),
            "confirmatory_eligible_row_count": sum(
                label in STRICT_LABELS for label in labels
            ),
            "normalized_emitted_artists": sorted(
                {
                    str(
                        row.get("normalized_artist")
                        or normalize_name(str(row.get("artist", "")))
                    )
                    for row in cluster_rows
                }
            ),
            "strict_conflict_references": [
                references[key] for key in sorted(references)
            ],
        }
    return summaries


def apply_title_cluster_summaries(rows: list[dict[str, Any]]) -> None:
    summaries = summarize_title_clusters(rows)
    for row in rows:
        title_key = str(row.get("normalized_title") or normalize_name(str(row.get("title", ""))))
        row["normalized_title"] = title_key
        row["catalog_title_cluster"] = summaries[title_key]


def request_cache_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for source_row in rows:
        row = dict(source_row)
        if (
            row.get("request_status") == "ok"
            and row.get("request_url")
            and row.get("raw_response_body") is not None
        ):
            cache[str(row["request_url"])] = row
    return cache


def load_request_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return request_cache_from_rows(read_jsonl(path))


def _cache_rows(cache: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [cache[url] for url in sorted(cache)]


SOURCE_ROW_FIELDS = (
    "record_type",
    "record_id",
    "generation_id",
    "context_id",
    "seed",
    "rank",
    "title",
    "artist",
    "reason",
    "batch_mode",
    "prompt_template_id",
    "model_id",
    "model_revision",
    "protocol_id",
    "protocol_hashes",
    "generation_row_sha256",
)


def _row_identity(row: Mapping[str, Any]) -> str:
    source = {field: row.get(field) for field in SOURCE_ROW_FIELDS}
    payload = json.dumps(
        source, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_resume_rows(
    input_rows: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> None:
    if len(input_rows) != len(checkpoint_rows):
        raise ValueError("checkpoint row count does not match input")
    for index, (input_row, checkpoint_row) in enumerate(
        zip(input_rows, checkpoint_rows)
    ):
        if _row_identity(input_row) != _row_identity(checkpoint_row):
            raise ValueError(f"checkpoint row {index} does not match input")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-archive", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--sleep-seconds", type=float, default=1.1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--retry-errors-only",
        action="store_true",
        help="re-query only rows whose Phase 2 catalog label is error",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore an existing output checkpoint",
    )
    parser.add_argument(
        "--reclassify-all",
        action="store_true",
        help="reclassify every input row under the current verifier version",
    )
    return parser


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    fetcher: Callable[[str], Any] | None = None,
    now: Callable[[], str] = utc_timestamp,
) -> int:
    args = _argument_parser().parse_args(argv)
    if args.offset < 0:
        raise ValueError("offset must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise ValueError("limit must be non-negative")
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")

    input_rows = read_jsonl(args.input)
    resuming = args.output.exists() and not args.no_resume
    if resuming:
        rows = read_jsonl(args.output)
        _validate_resume_rows(input_rows, rows)
    else:
        rows = input_rows

    if args.no_resume:
        archive: list[dict[str, Any]] = []
        write_jsonl(args.evidence_archive, [])
    else:
        archive = (
            read_jsonl(args.evidence_archive)
            if args.evidence_archive.exists()
            else []
        )
    cache = {} if args.no_resume else load_request_cache(args.cache)
    if not args.no_resume:
        # The append-only archive is authoritative over any older cache snapshot.
        cache.update(request_cache_from_rows(archive))
    flushed_archive_count = len(archive)

    # Establish the row checkpoint before the first network call. Evidence from a
    # first-row interruption can then be reused without erasing or re-querying it.
    if not resuming:
        apply_title_cluster_summaries(rows)
        write_jsonl(args.output, rows)

    def append_new_evidence() -> None:
        nonlocal flushed_archive_count
        new_rows = archive[flushed_archive_count:]
        append_jsonl(args.evidence_archive, new_rows)
        flushed_archive_count += len(new_rows)

    def checkpoint_requests() -> None:
        append_new_evidence()

    effective_fetcher = fetcher
    if effective_fetcher is None:
        effective_fetcher = lambda url: request_raw(url, timeout=args.timeout)
    session = CatalogRequestSession(
        fetcher=effective_fetcher,
        cache=cache,
        archive=archive,
        now=now,
        sleep_seconds=args.sleep_seconds,
        max_attempts=args.max_attempts,
        on_record=append_new_evidence,
    )

    selected_indices: list[int] = []
    for index, row in enumerate(rows):
        label = row.get("phase2_catalog_label")
        if args.reclassify_all:
            selected_indices.append(index)
        elif args.retry_errors_only:
            if label == "error":
                selected_indices.append(index)
        elif label not in FINAL_LABELS:
            selected_indices.append(index)
    selected_indices = selected_indices[args.offset :]
    if args.limit is not None:
        selected_indices = selected_indices[: args.limit]

    for index in selected_indices:
        row = rows[index]
        result = verify_pair(
            str(row.get("title", "")),
            str(row.get("artist", "")),
            accepted_artists=row.get("accepted_artists"),
            session=session,
        )
        row.update(result)
        apply_title_cluster_summaries(rows)
        write_jsonl(args.output, rows)
        checkpoint_requests()

    apply_title_cluster_summaries(rows)
    write_jsonl(args.output, rows)
    checkpoint_requests()
    if args.cache is not None:
        write_jsonl(args.cache, _cache_rows(cache))
    labels = Counter(
        str(row.get("phase2_catalog_label", "pending")) for row in rows
    )
    summary = {
        "input_row_count": len(rows),
        "processed_row_count": len(selected_indices),
        "normalized_title_cluster_count": len(summarize_title_clusters(rows)),
        "catalog_label_counts": dict(sorted(labels.items())),
        "evidence_request_count": len(archive),
        "cache_entry_count": len(cache),
    }
    print("PHASE2_CATALOG_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
    return 0


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
