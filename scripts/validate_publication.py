"""Validate the public report, artifacts, figures, and local links."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from build_publication_manifest import build_manifest
from export_phase2_mechanism_intervention_probe import select_catalog_clusters


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_MARKDOWN_PATHS = (
    Path("README.md"),
    Path("README.zh-CN.md"),
    Path("reports/music_relation_hallucination_technical_report.md"),
    Path("reports/music_relation_hallucination_summary.zh.md"),
    Path("reports/publication_readiness_audit.md"),
    Path("docs/reproduce_publication.md"),
    Path("reports/phase2_granite_confirmatory_catalog_yield.md"),
    Path("reports/phase2_granite_confirmatory_catalog_yield.zh.md"),
    Path("reports/related_work_positioning_2026_07.zh.md"),
    Path("reports/autoresearchclaw_literature_audit_2026_07.zh.md"),
    Path("reports/academic_research_suite_literature_review_2026_07.zh.md"),
    Path("docs/reproduce_phase2_catalog_yield.md"),
    Path("docs/public_data_policy.md"),
    Path("LICENSING.md"),
    Path("THIRD_PARTY_NOTICES.md"),
    Path("CONTRIBUTING.md"),
)

REQUIRED_PATHS = (
    *PUBLIC_MARKDOWN_PATHS,
    Path(".github/workflows/validate.yml"),
    Path("CITATION.cff"),
    Path("LICENSE"),
    Path("NOTICE"),
    Path("docs/independent_holdout_protocol.md"),
    Path("reports/hf_jobs_run_log.md"),
    Path("reports/figures/holdout_verifier_metrics.svg"),
    Path("reports/figures/causal_trace_residual.svg"),
    Path("runs/publication_manifest.json"),
    Path("runs/hf_job_metadata.json"),
    Path("runs/independent_holdout_raw_generations.jsonl"),
    Path("runs/independent_holdout_catalog_verified.jsonl"),
    Path("runs/independent_holdout_verifier_rows.jsonl"),
    Path("runs/independent_holdout_verifier_summary.json"),
    Path("runs/qwen3_4b_cross_model_verifier_rows.jsonl"),
    Path("runs/qwen3_4b_cross_model_verifier_summary.json"),
    Path("runs/holdout_title_contrast_rows.jsonl"),
    Path("runs/holdout_title_contrast_summary.json"),
    Path("runs/holdout_sequence_causal_trace_rows.jsonl"),
    Path("runs/holdout_sequence_causal_trace_summary.json"),
    Path("config/phase2_mechanism_intervention_protocol.json"),
    Path("docs/phase2_mechanism_intervention_protocol.md"),
    Path("runs/phase2_granite_combined_catalog_verified_v2.jsonl"),
    Path("runs/phase2_granite_final_selected_conflicts_v2.jsonl"),
    Path("runs/phase2_granite_final_selected_exact_v2.jsonl"),
    Path("runs/private_evidence_receipt.json"),
    Path("runs/phase2_granite_extension_catalog_stdout_v2_resume.log"),
    Path("runs/phase2_granite_extension_catalog_stderr_v2_resume.log"),
    Path("runs/phase2_granite_extension_catalog_stdout_v2_retry1.log"),
    Path("runs/phase2_granite_extension_catalog_stderr_v2_retry1.log"),
    Path("runs/phase2_granite_extension_catalog_stdout_v2_retry2.log"),
    Path("runs/phase2_granite_extension_catalog_stderr_v2_retry2.log"),
    Path("runs/phase2_granite_final_catalog_summary_v2.json"),
    Path("runs/phase2_granite_final_catalog_receipt_v2.json"),
    Path("runs/autoresearchclaw_literature_review_receipt_20260714.json"),
    Path("runs/academic_research_suite_literature_matrix_20260714.json"),
)

MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_literature_matrix(project_root: Path) -> tuple[int, int]:
    matrix = load_json(
        project_root / "runs/academic_research_suite_literature_matrix_20260714.json"
    )
    if matrix.get("review_type") != "targeted_structured_literature_review":
        raise ValueError("literature matrix review type is missing or overstated")

    included = matrix.get("included")
    excluded = matrix.get("excluded")
    synthesis = matrix.get("synthesis", {})
    if not isinstance(included, list) or not isinstance(excluded, list):
        raise ValueError("literature matrix records must be lists")
    if synthesis.get("included_count") != len(included):
        raise ValueError("literature matrix included count is stale")
    if synthesis.get("excluded_count") != len(excluded):
        raise ValueError("literature matrix excluded count is stale")

    ids = [row.get("id") for row in included]
    if not all(isinstance(record_id, str) and record_id for record_id in ids):
        raise ValueError("literature matrix has a missing included-record id")
    if len(ids) != len(set(ids)):
        raise ValueError("literature matrix has duplicate included-record ids")
    if not all(row.get("evidence_grade") in {"A", "B", "C"} for row in included):
        raise ValueError("literature matrix has an invalid evidence grade")
    if not all(str(row.get("url", "")).startswith("https://") for row in included):
        raise ValueError("literature matrix included URLs must use HTTPS")
    if not all(row.get("verification_level") for row in included):
        raise ValueError("literature matrix has an unverified included record")

    closest = synthesis.get("closest_prior_work", [])
    if not set(closest).issubset(set(ids)):
        raise ValueError("literature matrix closest-prior ids are not included")
    return len(included), len(excluded)


def find_broken_local_links(markdown_path: Path) -> list[str]:
    text = markdown_path.read_text(encoding="utf-8")
    broken: list[str] = []
    for raw_target in MARKDOWN_LINK_PATTERN.findall(text):
        target = raw_target.strip().strip("<>")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path_part = unquote(target.split("#", 1)[0])
        if not path_part:
            continue
        resolved = (markdown_path.parent / path_part).resolve()
        if not resolved.exists():
            broken.append(target)
    return broken


def _require_fragment(text: str, fragment: str, label: str) -> None:
    if fragment not in text:
        raise ValueError(f"{label} is missing expected fragment: {fragment}")


def _validate_report_metrics(project_root: Path) -> None:
    english = (
        project_root / "reports/music_relation_hallucination_technical_report.md"
    ).read_text(encoding="utf-8")
    chinese = (
        project_root / "reports/music_relation_hallucination_summary.zh.md"
    ).read_text(encoding="utf-8")
    qwen_17 = load_json(
        project_root / "runs/independent_holdout_verifier_summary.json"
    )
    qwen_4 = load_json(
        project_root / "runs/qwen3_4b_cross_model_verifier_summary.json"
    )
    qwen_17_rows = load_jsonl(
        project_root / "runs/independent_holdout_verifier_rows.jsonl"
    )
    qwen_4_rows = load_jsonl(
        project_root / "runs/qwen3_4b_cross_model_verifier_rows.jsonl"
    )
    causal = load_json(
        project_root / "runs/holdout_sequence_causal_trace_summary.json"
    )
    reason_swap = load_json(
        project_root / "runs/qwen_scope_music_reason_swap_pilot_summary.json"
    )
    vocality_reason = load_json(
        project_root / "runs/controlled_vocality_reason_summary.json"
    )
    vocality_patch = load_json(
        project_root / "runs/controlled_vocality_path_patching_summary.json"
    )
    vocality_fields = load_json(
        project_root / "runs/controlled_vocality_field_probe_summary.json"
    )

    q17_choice = qwen_17["choice_metrics"]
    q17_primary = qwen_17["primary_metrics"]
    q4_choice = qwen_4["choice_metrics"]
    q4_sequence = qwen_4["catalog_sequence_metrics"]
    expected_fragments = (
        (
            english,
            "| Independent choice | "
            f"**{q17_choice['exact_correct']}/{q17_choice['exact_count']}** | "
            f"**{q17_choice['conflict_correct']}/{q17_choice['conflict_count']}** |",
            "English report",
        ),
        (
            english,
            "| Frozen OR | "
            f"{q17_primary['exact_correct']}/{q17_primary['exact_count']} | "
            f"{q17_primary['conflict_correct']}/{q17_primary['conflict_count']} |",
            "English report",
        ),
        (
            chinese,
            "| 独立 A/B | "
            f"**{q17_choice['exact_correct']}/{q17_choice['exact_count']}** | "
            f"**{q17_choice['conflict_correct']}/{q17_choice['conflict_count']}** |",
            "Chinese summary",
        ),
        (
            chinese,
            "| 独立 A/B | "
            f"{q4_choice['exact_correct']}/{q4_choice['exact_count']} | "
            f"{q4_choice['conflict_correct']}/{q4_choice['conflict_count']} |",
            "Chinese summary",
        ),
        (
            english,
            "| Factual complete artist | "
            f"**{q4_sequence['exact_correct']}/{q4_sequence['exact_count']}** | "
            f"{q4_sequence['conflict_correct']}/{q4_sequence['conflict_count']} |",
            "English report",
        ),
        (
            english,
            f"maximum counterfactual-effect reproduction error: "
            f"`{causal['maximum_contrast_reproduction_error']:.5f}`",
            "English report",
        ),
        (
            english,
            f"own reason beat opposite-context reason in "
            f"{round(reason_swap['overall']['own_beats_opposite_context_rate'] * reason_swap['record_count'])}/"
            f"{reason_swap['record_count']} pairs",
            "English report exploratory evidence",
        ),
        (
            english,
            f"correct for {round(vocality_reason['pair_overall']['matched_direction_accuracy'] * vocality_reason['record_count'])}/"
            f"{vocality_reason['record_count']} pairs",
            "English report exploratory evidence",
        ),
        (
            english,
            f"Layer-18 attention recovered `{next(point['mean_recovery'] for point in vocality_patch['choice_component_curve'] if point['layer'] == 18 and point['component'] == 'attention'):.3f}` "
            f"of the closed-set choice effect and `{next(point['mean_recovery'] for point in vocality_patch['pair_component_curve'] if point['layer'] == 18 and point['component'] == 'attention'):.3f}`",
            "English report exploratory evidence",
        ),
        (
            english,
            f"correct artist effect (`{vocality_fields['field_effects']['instrumental_awake']['artist_mean_logp']:+.3f}`)",
            "English report exploratory evidence",
        ),
    )
    for text, fragment, label in expected_fragments:
        _require_fragment(text, fragment, label)

    def unique_choice_misses(
        rows: list[dict[str, Any]],
    ) -> set[tuple[str, str, str]]:
        return {
            (row["title"], row["emitted_artist"], row["reference_artist"])
            for row in rows
            if row["catalog_label"] == "catalog_conflict"
            and not row["choice_predicts_conflict"]
        }

    qwen_17_misses = unique_choice_misses(qwen_17_rows)
    qwen_4_misses = unique_choice_misses(qwen_4_rows)
    if qwen_17_misses != qwen_4_misses or len(qwen_17_misses) != 3:
        raise ValueError(
            "the same-three-unique-conflicts replication claim no longer holds"
        )
    for title, emitted_artist, reference_artist in qwen_17_misses:
        _require_fragment(english, title, "English report")
        _require_fragment(english, emitted_artist, "English report")
        _require_fragment(english, reference_artist, "English report")


def _validate_row_counts(project_root: Path) -> None:
    pairs = (
        (
            "independent_holdout_verifier",
            "independent_holdout_verifier_rows.jsonl",
            "independent_holdout_verifier_summary.json",
        ),
        (
            "qwen3_4b_cross_model_verifier",
            "qwen3_4b_cross_model_verifier_rows.jsonl",
            "qwen3_4b_cross_model_verifier_summary.json",
        ),
        (
            "holdout_title_contrast",
            "holdout_title_contrast_rows.jsonl",
            "holdout_title_contrast_summary.json",
        ),
        (
            "holdout_sequence_causal_trace",
            "holdout_sequence_causal_trace_rows.jsonl",
            "holdout_sequence_causal_trace_summary.json",
        ),
    )
    for label, rows_name, summary_name in pairs:
        rows = load_jsonl(project_root / "runs" / rows_name)
        summary = load_json(project_root / "runs" / summary_name)
        if len(rows) != summary["row_count"]:
            raise ValueError(
                f"{label} row count mismatch: {len(rows)} != {summary['row_count']}"
            )
        if not summary["technical_gate"]:
            raise ValueError(f"{label} technical gate is false")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_phase2(project_root: Path) -> None:
    summary_path = project_root / "runs/phase2_granite_final_catalog_summary_v2.json"
    receipt_path = project_root / "runs/phase2_granite_final_catalog_receipt_v2.json"
    summary = load_json(summary_path)
    receipt = load_json(receipt_path)
    expected = {
        "combined_row_count": 385,
        "combined_unique_normalized_title_count": 304,
        "selected_conflict_cluster_count": 7,
        "selected_exact_cluster_count": 25,
        "minimum_conflict_clusters_to_continue": 30,
        "decision": "STOP_INSUFFICIENT_STRICT_CONFLICT_CLUSTERS",
        "formal_mechanism_run_allowed": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"Phase 2 final summary mismatch for {key}")
    if summary["catalog_validation"]["strict_catalog_row_count"] != 50:
        raise ValueError("Phase 2 strict replay count changed")
    if summary["combined_catalog_label_counts"] != {
        "ambiguous": 13,
        "excluded": 322,
        "strict_conflict": 9,
        "strict_exact": 41,
    }:
        raise ValueError("Phase 2 final catalog labels changed")

    combined_rows = load_jsonl(
        project_root / "runs/phase2_granite_combined_catalog_verified_v2.jsonl"
    )
    conflicts = load_jsonl(
        project_root / "runs/phase2_granite_final_selected_conflicts_v2.jsonl"
    )
    exact = load_jsonl(
        project_root / "runs/phase2_granite_final_selected_exact_v2.jsonl"
    )
    if (len(combined_rows), len(conflicts), len(exact)) != (385, 7, 25):
        raise ValueError("Phase 2 released row counts changed")
    protocol = load_json(
        project_root / "config/phase2_mechanism_intervention_protocol.json"
    )
    replay_conflicts, replay_exact, replay_selection = select_catalog_clusters(
        combined_rows, protocol
    )
    if replay_conflicts != conflicts or replay_exact != exact:
        raise ValueError("Phase 2 public derived-row selection replay changed")
    if (
        replay_selection["selected_conflict_cluster_count"] != 7
        or replay_selection["selected_exact_cluster_count"] != 25
    ):
        raise ValueError("Phase 2 public stop-gate replay changed")

    if _sha256(summary_path) != receipt["assets"]["final_summary"]["sha256"]:
        raise ValueError("Phase 2 final summary hash does not match its receipt")
    verifier = project_root / "scripts/verify_phase2_catalog.py"
    if _sha256(verifier) != receipt["catalog_verifier"]["sha256"]:
        raise ValueError("Phase 2 catalog verifier changed after the frozen run")

    private_evidence = load_json(
        project_root / "runs/private_evidence_receipt.json"
    )
    omitted = {
        row["path"]: row for row in private_evidence["omitted_assets"]
    }
    combined_path = "runs/phase2_granite_combined_catalog_evidence_v2.jsonl.gz"
    combined = omitted.get(combined_path)
    if not combined:
        raise ValueError("Phase 2 private-evidence receipt is missing")
    if (
        combined["sha256"]
        != receipt["assets"]["combined_evidence_gzip"]["sha256"]
        or combined["decompressed_sha256"]
        != receipt["evidence_merge"]["combined_raw_sha256"]
        or combined["decompressed_bytes"]
        != receipt["evidence_merge"]["combined_raw_bytes"]
    ):
        raise ValueError("Phase 2 private-evidence receipt changed")
    for relative_path in omitted:
        if (project_root / relative_path).exists():
            raise ValueError(f"raw third-party response archive is public: {relative_path}")

    recovery_steps = (
        ("resume", 9, 83),
        ("retry1", 8, 9),
        ("retry2", 0, 8),
    )
    for suffix, expected_errors, expected_processed in recovery_steps:
        stdout_path = (
            project_root
            / f"runs/phase2_granite_extension_catalog_stdout_v2_{suffix}.log"
        )
        marker = "PHASE2_CATALOG_SUMMARY_JSON="
        lines = [
            line
            for line in stdout_path.read_text(encoding="utf-8").splitlines()
            if line.startswith(marker)
        ]
        if len(lines) != 1:
            raise ValueError(f"Phase 2 recovery marker mismatch for {suffix}")
        recovery = json.loads(lines[0][len(marker) :])
        if recovery["catalog_label_counts"].get("error", 0) != expected_errors:
            raise ValueError(f"Phase 2 recovery error count changed for {suffix}")
        if recovery["processed_row_count"] != expected_processed:
            raise ValueError(
                f"Phase 2 recovery processed-row count changed for {suffix}"
            )
        stderr_path = (
            project_root
            / f"runs/phase2_granite_extension_catalog_stderr_v2_{suffix}.log"
        )
        if stderr_path.stat().st_size != 0:
            raise ValueError(f"Phase 2 recovery stderr is nonempty for {suffix}")

    english = (
        project_root / "reports/phase2_granite_confirmatory_catalog_yield.md"
    ).read_text(encoding="utf-8")
    chinese = (
        project_root / "reports/phase2_granite_confirmatory_catalog_yield.zh.md"
    ).read_text(encoding="utf-8")
    for text, fragment, label in (
        (
            english,
            "only 7 unique strict-conflict title clusters",
            "Phase 2 English report",
        ),
        (
            english,
            "This is a catalog-yield boundary result",
            "Phase 2 English report",
        ),
        (chinese, "只得到 **7 个", "Phase 2 Chinese report"),
        (chinese, "H1/H2 根本没有被测试", "Phase 2 Chinese report"),
    ):
        _require_fragment(text, fragment, label)


def _validate_figures(project_root: Path) -> None:
    for relative_path in (
        Path("reports/figures/holdout_verifier_metrics.svg"),
        Path("reports/figures/causal_trace_residual.svg"),
    ):
        root = ET.parse(project_root / relative_path).getroot()
        if not root.tag.endswith("svg"):
            raise ValueError(f"{relative_path.as_posix()} is not an SVG")
        width = float(root.attrib.get("width", 0))
        height = float(root.attrib.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"{relative_path.as_posix()} has invalid dimensions")

        for element in root.iter():
            if element.tag.endswith("polyline"):
                for pair in element.attrib.get("points", "").split():
                    x_text, y_text = pair.split(",", 1)
                    x, y = float(x_text), float(y_text)
                    if not (0 <= x <= width and 0 <= y <= height):
                        raise ValueError(
                            f"{relative_path.as_posix()} has an out-of-bounds point"
                        )
            if not element.tag.endswith("text"):
                continue
            value = "".join(element.itertext())
            x = float(element.attrib["x"])
            y = float(element.attrib["y"])
            size = float(element.attrib.get("font-size", 14))
            estimated_length = len(value) * size * 0.62
            anchor = element.attrib.get("text-anchor", "start")
            rotated = "rotate(-90" in element.attrib.get("transform", "")
            if rotated:
                left, right = x - size * 0.6, x + size * 0.6
                top, bottom = y - estimated_length / 2, y + estimated_length / 2
            else:
                left = x - estimated_length if anchor == "end" else x
                if anchor == "middle":
                    left = x - estimated_length / 2
                right = left + estimated_length
                top, bottom = y - size, y + size * 0.3
            if left < -2 or right > width + 2 or top < -2 or bottom > height + 2:
                raise ValueError(
                    f"{relative_path.as_posix()} has estimated text overflow: {value}"
                )

    causal_svg = (
        project_root / "reports/figures/causal_trace_residual.svg"
    ).read_text(encoding="utf-8")
    _require_fragment(
        causal_svg,
        'transform="rotate(-90.0',
        "Causal figure y-axis label",
    )


def validate_publication(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    missing = [
        path.as_posix() for path in REQUIRED_PATHS if not (project_root / path).exists()
    ]
    if missing:
        raise ValueError(f"missing publication files: {missing}")

    broken_links: dict[str, list[str]] = {}
    for relative_path in PUBLIC_MARKDOWN_PATHS:
        broken = find_broken_local_links(project_root / relative_path)
        if broken:
            broken_links[relative_path.as_posix()] = broken
    if broken_links:
        raise ValueError(f"broken local links: {broken_links}")

    for relative_path in PUBLIC_MARKDOWN_PATHS:
        text = (project_root / relative_path).read_text(encoding="utf-8")
        if "\ufffd" in text:
            raise ValueError(f"replacement character found in {relative_path}")

    active_claim_text = "\n".join(
        (project_root / path).read_text(encoding="utf-8")
        for path in (
            Path("README.md"),
            Path("reports/music_relation_hallucination_technical_report.md"),
            Path("reports/phase2_granite_confirmatory_catalog_yield.md"),
        )
    ).casefold()
    if "preregistered warning rule" in active_claim_text:
        raise ValueError(
            "the active report must say pre-specified because no external "
            "registration timestamp exists"
        )
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    if "This private repository" in readme:
        raise ValueError("public README still describes the repository as private")
    if "contrib/music-preference-lens" in readme:
        raise ValueError("public README still contains the old monorepo path")

    stored_manifest = load_json(project_root / "runs/publication_manifest.json")
    expected_manifest = build_manifest()
    if stored_manifest != expected_manifest:
        raise ValueError(
            "runs/publication_manifest.json is stale; rebuild it before publishing"
        )
    if stored_manifest["protocol_status"] != {
        "label": "pre_specified",
        "frozen_before_scoring": True,
        "externally_timestamped": False,
    }:
        raise ValueError("protocol registration status is missing or overstated")

    job_metadata = load_json(project_root / "runs/hf_job_metadata.json")
    summary_markers = {
        "holdout_generation": "HOLDOUT_GENERATION_SUMMARY_JSON=",
        "qwen3_1_7b_confirmation": "HOLDOUT_VERIFY_SUMMARY_JSON=",
        "qwen3_4b_cross_model": "HOLDOUT_VERIFY_SUMMARY_JSON=",
        "title_counterfactual_retry": "TITLE_CONTRAST_SUMMARY_JSON=",
        "causal_trace_artifact_retry": "SEQ_CAUSAL_ARTIFACT_CHUNK_JSON=",
    }
    for job in job_metadata["jobs"]:
        log_text = (project_root / job["terminal_log_path"]).read_text(
            encoding="utf-8"
        )
        if summary_markers[job["role"]] not in log_text:
            raise ValueError(f"expected terminal marker missing: {job['job_id']}")

    literature_included, literature_excluded = _validate_literature_matrix(project_root)
    _validate_report_metrics(project_root)
    _validate_row_counts(project_root)
    _validate_phase2(project_root)
    _validate_figures(project_root)
    return {
        "status": "ready",
        "markdown_files_checked": len(PUBLIC_MARKDOWN_PATHS),
        "required_files_checked": len(REQUIRED_PATHS),
        "publication_files_hashed": stored_manifest["publication_file_count"],
        "local_link_failures": 0,
        "manifest_version": stored_manifest["manifest_version"],
        "technical_gates_checked": 11,
        "publication_technical_gates_checked": 5,
        "exploratory_technical_gates_checked": 6,
        "hf_jobs_archived": len(job_metadata["jobs"]),
        "phase2_hf_jobs_recorded": 3,
        "literature_sources_checked": literature_included,
        "literature_exclusions_checked": literature_excluded,
    }


def main() -> int:
    print(json.dumps(validate_publication(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
