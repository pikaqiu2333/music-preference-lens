"""Build a machine-readable manifest for the public technical report."""

from __future__ import annotations

import base64
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from export_independent_holdout_verifier_probe import select_holdout_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in (
            "primary_metrics",
            "choice_metrics",
            "catalog_sequence_metrics",
            "catalog_first_token_metrics",
            "generation_first_token_metrics",
            "generation_sequence_metrics",
            "confirmation_status",
            "technical_gate",
        )
    }


def validate_job_archive(job_metadata: dict[str, Any]) -> list[Path]:
    paths = [Path("runs/hf_job_metadata.json")]
    for job in job_metadata["jobs"]:
        if job["status"] != "COMPLETED":
            raise ValueError(f"archived Job is not complete: {job['job_id']}")
        log_path = Path(job["terminal_log_path"])
        if sha256(PROJECT_ROOT / log_path) != job["terminal_log_sha256"]:
            raise ValueError(f"terminal log hash mismatch: {job['job_id']}")
        if (PROJECT_ROOT / log_path).stat().st_size != job["terminal_log_bytes"]:
            raise ValueError(f"terminal log size mismatch: {job['job_id']}")

        exact_path = Path(job["embedded_script_exact_base64_path"])
        payload = base64.b64decode(
            (PROJECT_ROOT / exact_path).read_text(encoding="ascii").strip(),
            validate=True,
        )
        if hashlib.sha256(payload).hexdigest() != job["embedded_script_sha256"]:
            raise ValueError(f"embedded Job script hash mismatch: {job['job_id']}")
        if len(payload) != job["embedded_script_bytes"]:
            raise ValueError(f"embedded Job script size mismatch: {job['job_id']}")
        readable_path = Path(job["embedded_script_readable_path"])
        if not (PROJECT_ROOT / readable_path).exists():
            raise ValueError(f"readable Job script is missing: {job['job_id']}")
        paths.extend((log_path, exact_path, readable_path))
    return paths


def build_manifest() -> dict[str, Any]:
    reason_order = load_json(
        PROJECT_ROOT / "runs" / "qwen_scope_music_reason_order_summary.json"
    )
    reason_swap = load_json(
        PROJECT_ROOT / "runs" / "qwen_scope_music_reason_swap_pilot_summary.json"
    )
    vocality_reason = load_json(
        PROJECT_ROOT / "runs" / "controlled_vocality_reason_summary.json"
    )
    vocality_patch = load_json(
        PROJECT_ROOT / "runs" / "controlled_vocality_path_patching_summary.json"
    )
    vocality_heads = load_json(
        PROJECT_ROOT / "runs" / "controlled_vocality_attention_head_summary.json"
    )
    vocality_fields = load_json(
        PROJECT_ROOT / "runs" / "controlled_vocality_field_probe_summary.json"
    )
    generation = load_json(
        PROJECT_ROOT / "runs" / "independent_holdout_generation_summary.json"
    )
    parsed = load_json(PROJECT_ROOT / "runs" / "independent_holdout_parse_summary.json")
    catalog_rows = load_jsonl(
        PROJECT_ROOT / "runs" / "independent_holdout_catalog_verified.jsonl"
    )
    verifier_bundle = load_json(
        PROJECT_ROOT / "runs" / "independent_holdout_verifier_bundle.json"
    )
    qwen_17 = load_json(
        PROJECT_ROOT / "runs" / "independent_holdout_verifier_summary.json"
    )
    qwen_4 = load_json(
        PROJECT_ROOT / "runs" / "qwen3_4b_cross_model_verifier_summary.json"
    )
    title = load_json(PROJECT_ROOT / "runs" / "holdout_title_contrast_summary.json")
    causal = load_json(
        PROJECT_ROOT / "runs" / "holdout_sequence_causal_trace_summary.json"
    )
    phase2_final = load_json(
        PROJECT_ROOT / "runs" / "phase2_granite_final_catalog_summary_v2.json"
    )
    phase2_receipt = load_json(
        PROJECT_ROOT / "runs" / "phase2_granite_final_catalog_receipt_v2.json"
    )
    phase3_catalog_receipt = load_json(
        PROJECT_ROOT / "runs" / "phase3_natural_pilot_catalog_receipt.json"
    )
    private_evidence = load_json(
        PROJECT_ROOT / "runs" / "private_evidence_receipt.json"
    )
    phase3_recoverability = load_json(
        PROJECT_ROOT / "runs" / "phase3_relation_recoverability_summary.json"
    )
    job_metadata = load_json(PROJECT_ROOT / "runs" / "hf_job_metadata.json")
    if not all(
        summary["technical_gate"] for summary in (qwen_17, qwen_4, title, causal)
    ):
        raise ValueError("a publication-stage technical gate is false")
    if not all(
        summary["technical_gate"]
        for summary in (
            reason_order,
            reason_swap,
            vocality_reason,
            vocality_patch,
            vocality_heads,
            vocality_fields,
        )
    ):
        raise ValueError("an integrated exploratory-evidence gate is false")
    if qwen_17["confirmation_status"] != "not_confirmed":
        raise ValueError("the pre-specified 1.7B result must remain negative")
    if qwen_4["confirmation_status"] != "not_confirmed":
        raise ValueError("the 4B cross-model result must remain negative")
    if (
        phase2_final["decision"]
        != "STOP_INSUFFICIENT_STRICT_CONFLICT_CLUSTERS"
        or phase2_final["formal_mechanism_run_allowed"]
        or phase2_final["selected_conflict_cluster_count"] != 7
    ):
        raise ValueError("the Phase 2 preregistered catalog-yield stop changed")
    if phase2_receipt["decision"] != phase2_final["decision"]:
        raise ValueError("the Phase 2 receipt and final summary disagree")

    omitted_by_path = {
        row["path"]: row for row in private_evidence["omitted_assets"]
    }
    expected_omitted_hashes = {
        "runs/phase2_granite_primary_catalog_evidence_v2.jsonl.gz": phase2_receipt[
            "assets"
        ]["primary_evidence_gzip"]["sha256"],
        "runs/phase2_granite_extension_catalog_evidence_v2.jsonl.gz": phase2_receipt[
            "assets"
        ]["extension_evidence_gzip"]["sha256"],
        "runs/phase2_granite_combined_catalog_evidence_v2.jsonl.gz": phase2_receipt[
            "assets"
        ]["combined_evidence_gzip"]["sha256"],
        "runs/phase3_natural_pilot_catalog_evidence.jsonl.gz": phase3_catalog_receipt[
            "catalog"
        ]["compressed_evidence_sha256"],
    }
    if set(omitted_by_path) != set(expected_omitted_hashes):
        raise ValueError("the public private-evidence receipt has stale paths")
    for relative_path, expected_hash in expected_omitted_hashes.items():
        if omitted_by_path[relative_path]["sha256"] != expected_hash:
            raise ValueError(f"private-evidence receipt hash mismatch: {relative_path}")
        if (PROJECT_ROOT / relative_path).exists():
            raise ValueError(f"raw third-party response archive is public: {relative_path}")
    if not (
        phase3_recoverability["technical_gate"]
        and phase3_recoverability["assay_validity_gate"]
    ):
        raise ValueError("the Phase 3 recoverability assay is invalid")
    if (
        phase3_recoverability["pilot_continuation_gate"]
        or phase3_recoverability["recoverable_relation_conflict_record_ids"]
        or phase3_recoverability["selected_conflict_cluster_count"] != 11
    ):
        raise ValueError("the Phase 3 preregistered recoverability stop changed")

    selected_rows, selection = select_holdout_rows(catalog_rows)
    selected_hashes = {row["selection_hash"] for row in selected_rows}
    bundle_hashes = {
        row["selection_hash"] for row in verifier_bundle["records"]
    }
    if selected_hashes != bundle_hashes:
        raise ValueError("verifier bundle does not match frozen catalog selection")
    catalog_label_counts = Counter(row["catalog_label"] for row in catalog_rows)

    jobs_by_role = {job["role"]: job for job in job_metadata["jobs"]}
    expected_job_roles = {
        "holdout_generation",
        "qwen3_1_7b_confirmation",
        "qwen3_4b_cross_model",
        "title_counterfactual_retry",
        "causal_trace_artifact_retry",
    }
    if set(jobs_by_role) != expected_job_roles:
        raise ValueError("archived Job roles do not match the publication stages")
    job_paths = validate_job_archive(job_metadata)

    publication_paths = [
        Path(".github/workflows/validate.yml"),
        Path(".gitignore"),
        Path("CITATION.cff"),
        Path("CONTRIBUTING.md"),
        Path("LICENSE"),
        Path("LICENSING.md"),
        Path("NOTICE"),
        Path("README.md"),
        Path("README.zh-CN.md"),
        Path("THIRD_PARTY_NOTICES.md"),
        Path("requirements-publication.txt"),
        Path("docs/independent_holdout_protocol.md"),
        Path("docs/public_data_policy.md"),
        Path("docs/reproduce_publication.md"),
        Path("docs/phase2_mechanism_intervention_protocol.md"),
        Path("docs/reproduce_phase2_catalog_yield.md"),
        Path("docs/phase3_natural_relation_discovery_protocol.md"),
        Path("docs/phase3_relation_recoverability_protocol.md"),
        Path("docs/reason_order_counterfactual_plan.md"),
        Path("docs/reason_swap_counterfactual_plan.md"),
        Path("docs/controlled_vocality_reason_plan.md"),
        Path("docs/controlled_vocality_path_patching_plan.md"),
        Path("docs/controlled_vocality_attention_head_plan.md"),
        Path("docs/controlled_vocality_field_diagnosis_plan.md"),
        Path("reports/music_relation_hallucination_technical_report.md"),
        Path("reports/music_relation_hallucination_summary.zh.md"),
        Path("reports/phase2_granite_confirmatory_catalog_yield.md"),
        Path("reports/phase2_granite_confirmatory_catalog_yield.zh.md"),
        Path("reports/phase3_qwen_relation_recoverability_pilot.zh.md"),
        Path("reports/related_work_positioning_2026_07.zh.md"),
        Path("reports/autoresearchclaw_literature_audit_2026_07.zh.md"),
        Path("reports/academic_research_suite_literature_review_2026_07.zh.md"),
        Path("reports/publication_readiness_audit.md"),
        Path("reports/independent_holdout_catalog_verification.md"),
        Path("reports/hf_jobs_run_log.md"),
        Path("reports/qwen_scope_music_reason_order_summary.zh.md"),
        Path("reports/qwen_scope_music_reason_swap_pilot_summary.zh.md"),
        Path("reports/controlled_vocality_reason_probe_summary.zh.md"),
        Path("reports/controlled_vocality_path_patching_summary.zh.md"),
        Path("reports/controlled_vocality_attention_head_summary.zh.md"),
        Path("reports/controlled_vocality_field_probe_summary.zh.md"),
        Path("reports/controlled_vocality_tracks_catalog_verification.md"),
        Path("reports/figures/holdout_verifier_metrics.svg"),
        Path("reports/figures/causal_trace_residual.svg"),
        Path("scripts/export_holdout_generation_probe.py"),
        Path("scripts/run_holdout_generation_probe.py"),
        Path("scripts/analyze_holdout_generations.py"),
        Path("scripts/verify_song_entity_catalog.py"),
        Path("scripts/export_independent_holdout_verifier_probe.py"),
        Path("scripts/run_independent_holdout_verifier_probe.py"),
        Path("scripts/export_holdout_title_contrast_probe.py"),
        Path("scripts/run_holdout_title_contrast_probe.py"),
        Path("scripts/export_holdout_sequence_causal_trace.py"),
        Path("scripts/run_holdout_sequence_causal_trace.py"),
        Path("scripts/build_embedded_hf_job.py"),
        Path("scripts/extract_job_log_artifacts.py"),
        Path("scripts/decode_zlib_job_artifact.py"),
        Path("scripts/export_reason_order_probe.py"),
        Path("scripts/run_reason_order_probe.py"),
        Path("scripts/analyze_reason_order_generations.py"),
        Path("scripts/export_reason_swap_probe.py"),
        Path("scripts/run_reason_swap_probe.py"),
        Path("scripts/export_controlled_vocality_reason_probe.py"),
        Path("scripts/run_controlled_vocality_reason_probe.py"),
        Path("scripts/export_controlled_vocality_path_patching_probe.py"),
        Path("scripts/run_controlled_vocality_path_patching_probe.py"),
        Path("scripts/export_controlled_vocality_attention_head_probe.py"),
        Path("scripts/run_controlled_vocality_attention_head_probe.py"),
        Path("scripts/export_controlled_vocality_field_probe.py"),
        Path("scripts/run_controlled_vocality_field_probe.py"),
        Path("scripts/render_technical_report_figures.py"),
        Path("scripts/finalize_phase2_catalog.py"),
        Path("scripts/merge_phase2_catalog_evidence.py"),
        Path("scripts/prepare_phase2_catalog_input.py"),
        Path("scripts/verify_phase2_catalog.py"),
        Path("scripts/export_phase3_natural_generation.py"),
        Path("scripts/run_phase3_natural_generation.py"),
        Path("scripts/prepare_phase3_catalog_input.py"),
        Path("scripts/export_phase3_relation_recoverability.py"),
        Path("scripts/run_phase3_relation_recoverability.py"),
        Path("scripts/extract_relation_knowledge_audit.py"),
        Path("scripts/build_publication_manifest.py"),
        Path("scripts/validate_publication.py"),
        Path("runs/independent_holdout_generation_bundle.json"),
        Path("runs/qwen_scope_music_reason_order_bundle.json"),
        Path("runs/qwen_scope_music_reason_order_raw_generations.jsonl"),
        Path("runs/qwen_scope_music_reason_order_generations.jsonl"),
        Path("runs/qwen_scope_music_reason_order_rows.jsonl"),
        Path("runs/qwen_scope_music_reason_order_summary.json"),
        Path("runs/qwen_scope_music_reason_order_original_parser_summary.json"),
        Path("runs/qwen_scope_music_reason_order_catalog_verified.jsonl"),
        Path("runs/qwen_scope_music_reason_order_catalog_summary.json"),
        Path("runs/qwen_scope_music_reason_swap_bundle.json"),
        Path("runs/qwen_scope_music_reason_swap_pilot_rows.jsonl"),
        Path("runs/qwen_scope_music_reason_swap_pilot_summary.json"),
        Path("runs/controlled_vocality_tracks_catalog_verified.jsonl"),
        Path("runs/controlled_vocality_reason_probe_bundle.json"),
        Path("runs/controlled_vocality_reason_pair_rows.jsonl"),
        Path("runs/controlled_vocality_reason_choice_rows.jsonl"),
        Path("runs/controlled_vocality_reason_summary.json"),
        Path("runs/controlled_vocality_path_patching_bundle.json"),
        Path("runs/controlled_vocality_path_patching_pair_rows.jsonl"),
        Path("runs/controlled_vocality_path_patching_choice_rows.jsonl"),
        Path("runs/controlled_vocality_path_patching_summary.json"),
        Path("runs/controlled_vocality_attention_head_bundle.json"),
        Path("runs/controlled_vocality_attention_head_pair_rows.jsonl"),
        Path("runs/controlled_vocality_attention_head_choice_rows.jsonl"),
        Path("runs/controlled_vocality_attention_head_summary.json"),
        Path("runs/controlled_vocality_field_probe_bundle.json"),
        Path("runs/controlled_vocality_field_probe_rows.jsonl"),
        Path("runs/controlled_vocality_field_probe_summary.json"),
        Path("runs/independent_holdout_raw_generations.jsonl"),
        Path("runs/independent_holdout_generation_summary.json"),
        Path("runs/independent_holdout_generated_pairs.jsonl"),
        Path("runs/independent_holdout_parse_summary.json"),
        Path("runs/independent_holdout_catalog_verified.jsonl"),
        Path("runs/independent_holdout_verifier_bundle.json"),
        Path("runs/independent_holdout_verifier_bundle.zlib.b64"),
        Path("runs/independent_holdout_verifier_rows.jsonl"),
        Path("runs/independent_holdout_verifier_summary.json"),
        Path("runs/qwen3_4b_cross_model_verifier_bundle.json"),
        Path("runs/qwen3_4b_cross_model_verifier_bundle.zlib.b64"),
        Path("runs/qwen3_4b_cross_model_verifier_rows.jsonl"),
        Path("runs/qwen3_4b_cross_model_verifier_summary.json"),
        Path("runs/holdout_title_contrast_bundle.json"),
        Path("runs/holdout_title_contrast_bundle.zlib.b64"),
        Path("runs/holdout_title_contrast_rows.jsonl"),
        Path("runs/holdout_title_contrast_summary.json"),
        Path("runs/holdout_sequence_causal_trace_bundle.json"),
        Path("runs/holdout_sequence_causal_trace_bundle.zlib.b64"),
        Path("runs/holdout_sequence_causal_trace_artifact.zlib.b64"),
        Path("runs/holdout_sequence_causal_trace_rows.jsonl"),
        Path("runs/holdout_sequence_causal_trace_summary.json"),
        Path("runs/jobs/run_holdout_generation_embedded.py"),
        Path("runs/jobs/run_holdout_verifier_embedded.py"),
        Path("runs/jobs/run_qwen3_4b_verifier_embedded.py"),
        Path("runs/jobs/run_title_contrast_embedded.py"),
        Path("runs/jobs/run_sequence_trace_embedded.py"),
        Path("runs/jobs/run_phase2_granite_smoke_generation_embedded.py"),
        Path("runs/jobs/run_phase2_granite_primary_generation_embedded.py"),
        Path("runs/jobs/run_phase2_granite_extension_generation_embedded.py"),
        Path("runs/phase2_granite_smoke_summary.json"),
        Path("runs/phase2_granite_primary_generation_bundle.json"),
        Path("runs/phase2_granite_primary_generation_artifact.json"),
        Path("runs/phase2_granite_extension_generation_bundle.json"),
        Path("runs/phase2_granite_extension_generation_artifact.json"),
        Path("runs/phase2_granite_extension_generation_hf.log"),
        Path("runs/phase2_granite_primary_catalog_verified_v2.jsonl"),
        Path("runs/phase2_granite_extension_catalog_verified_v2.jsonl"),
        Path("runs/phase2_granite_combined_catalog_verified_v2.jsonl"),
        Path("runs/phase2_granite_final_selected_conflicts_v2.jsonl"),
        Path("runs/phase2_granite_final_selected_exact_v2.jsonl"),
        Path("runs/phase2_granite_extension_catalog_stdout_v2_resume.log"),
        Path("runs/phase2_granite_extension_catalog_stderr_v2_resume.log"),
        Path("runs/phase2_granite_extension_catalog_stdout_v2_retry1.log"),
        Path("runs/phase2_granite_extension_catalog_stderr_v2_retry1.log"),
        Path("runs/phase2_granite_extension_catalog_stdout_v2_retry2.log"),
        Path("runs/phase2_granite_extension_catalog_stderr_v2_retry2.log"),
        Path("runs/phase2_granite_final_catalog_summary_v2.json"),
        Path("runs/phase2_granite_final_catalog_receipt_v2.json"),
        Path("runs/phase3_natural_pilot_generation_bundle.json"),
        Path("runs/phase3_natural_pilot_generation_artifact.json"),
        Path("runs/phase3_natural_pilot_generation_hf.log"),
        Path("runs/phase3_natural_pilot_generation_rows.jsonl"),
        Path("runs/phase3_natural_pilot_generation_summary.json"),
        Path("runs/phase3_natural_pilot_catalog_input.jsonl"),
        Path("runs/phase3_natural_pilot_catalog_input_summary.json"),
        Path("runs/phase3_natural_pilot_catalog_verified.jsonl"),
        Path("runs/phase3_natural_pilot_catalog_receipt.json"),
        Path("runs/private_evidence_receipt.json"),
        Path("runs/phase3_relation_recoverability_bundle.json"),
        Path("runs/phase3_relation_recoverability_artifact.json"),
        Path("runs/phase3_relation_recoverability_hf.log"),
        Path("runs/phase3_relation_recoverability_rows.jsonl"),
        Path("runs/phase3_relation_recoverability_summary.json"),
        Path("runs/jobs/run_phase3_natural_pilot_generation_embedded.py"),
        Path("runs/jobs/run_phase3_relation_recoverability_embedded.py"),
        Path("runs/autoresearchclaw_literature_review_receipt_20260714.json"),
        Path("runs/academic_research_suite_literature_matrix_20260714.json"),
        Path("tests/test_independent_holdout.py"),
        Path("tests/test_finalize_phase2_catalog.py"),
        Path("tests/test_merge_phase2_catalog_evidence.py"),
        Path("tests/test_phase2_catalog.py"),
        Path("tests/test_phase2_mechanism_intervention.py"),
        Path("tests/test_phase2_protocol.py"),
        Path("tests/test_phase3_natural_generation.py"),
        Path("tests/test_phase3_natural_relation_protocol.py"),
        Path("tests/test_prepare_phase3_catalog_input.py"),
        Path("tests/test_phase3_relation_recoverability.py"),
        Path("tests/test_publication_assets.py"),
        Path("config/phase2_mechanism_intervention_protocol.json"),
        Path("config/phase3_natural_relation_discovery_protocol.json"),
        Path("config/phase3_relation_recoverability_protocol.json"),
        *job_paths,
    ]
    publication_paths = sorted(set(publication_paths))
    return {
        "manifest_version": "music_relation_hallucination_public_v1",
        "date": "2026-07-18",
        "scope": "open-ended music title-artist relation hallucination",
        "public_distribution": {
            "snapshot": "public_v1",
            "raw_third_party_response_bodies_included": False,
            "private_evidence_receipt": "runs/private_evidence_receipt.json",
            "omitted_asset_count": len(private_evidence["omitted_assets"]),
            "claim_boundary": private_evidence["claim_boundary"],
        },
        "models": {
            "generator_and_same_model_verifier": {
                "model_id": "Qwen/Qwen3-1.7B-Base",
                **job_metadata["model_revisions"]["Qwen/Qwen3-1.7B-Base"],
            },
            "cross_model_verifier": {
                "model_id": "Qwen/Qwen3-4B-Base",
                **job_metadata["model_revisions"]["Qwen/Qwen3-4B-Base"],
            },
            "phase2_confirmatory_generator": {
                "model_id": "ibm-granite/granite-4.1-3b-base",
                "revision": "dacb9cb9157bec98e99b09f285c92a4d58405c96",
            },
            "phase3_generator_and_recoverability_model": {
                "model_id": "Qwen/Qwen3-4B-Base",
                "revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
            },
        },
        "protocol_status": {
            "label": "pre_specified",
            "frozen_before_scoring": True,
            "externally_timestamped": False,
        },
        "phase2_protocol_status": {
            "label": "externally_preregistered",
            "frozen_before_generation": True,
            "externally_timestamped": True,
            "public_gist": "https://gist.github.com/a56668283b095f59f0eacf0527395b58",
        },
        "phase3_protocol_status": {
            "label": "frozen_before_pilot_catalog_results",
            "frozen_before_catalog_scoring": True,
            "externally_timestamped": False,
        },
        "integrated_exploratory_evidence": {
            "reason_order": {
                "matched_playlist_count": len(reason_order["matched_overlaps"]),
                "mean_exact_pair_jaccard": reason_order[
                    "mean_exact_pair_jaccard"
                ],
                "technical_gate": reason_order["technical_gate"],
            },
            "reason_swap": {
                "record_count": reason_swap["record_count"],
                "overall_own_beats_opposite_rate": reason_swap["overall"][
                    "own_beats_opposite_context_rate"
                ],
                "verified_exact_own_beats_same_rate": reason_swap[
                    "by_entity_group"
                ]["verified_exact"]["own_beats_same_context_swap_rate"],
                "verified_exact_own_beats_neutral_rate": reason_swap[
                    "by_entity_group"
                ]["verified_exact"]["own_beats_neutral_rate"],
                "technical_gate": reason_swap["technical_gate"],
            },
            "controlled_vocality_behavior": {
                "record_count": vocality_reason["record_count"],
                "matched_direction_accuracy": vocality_reason["pair_overall"][
                    "matched_direction_accuracy"
                ],
                "technical_gate": vocality_reason["technical_gate"],
            },
            "controlled_vocality_mechanism": {
                "layer18_attention_pair_mean_recovery": next(
                    point["mean_recovery"]
                    for point in vocality_patch["pair_component_curve"]
                    if point["layer"] == 18 and point["component"] == "attention"
                ),
                "layer18_attention_choice_mean_recovery": next(
                    point["mean_recovery"]
                    for point in vocality_patch["choice_component_curve"]
                    if point["layer"] == 18 and point["component"] == "attention"
                ),
                "selected_heads": vocality_fields["selected_heads"],
                "head_target_layer": vocality_heads["target_layer"],
                "technical_gates": {
                    "component": vocality_patch["technical_gate"],
                    "head": vocality_heads["technical_gate"],
                    "field": vocality_fields["technical_gate"],
                },
            },
            "controlled_vocality_field_effects": vocality_fields["field_effects"],
        },
        "independent_holdout": {
            "generation_count": generation["generation_count"],
            "generation_technical_gate": generation["technical_gate"],
            "parsed_generation_count": parsed["parsed_generation_count"],
            "parsed_pair_count": parsed["parsed_pair_count"],
            "parser_technical_gate": parsed["technical_gate"],
            "broad_catalog_label_counts": dict(catalog_label_counts),
            "strict_exact_pool_count": selection[
                "double_source_exact_pool_count"
            ],
            "strict_conflict_pool_count": selection[
                "unique_shared_conflict_pool_count"
            ],
            "selected_per_label": selection["selected_per_label"],
            "selection_hash_salt": selection["hash_salt"],
            "unique_selected_relations": {
                "exact": qwen_17["unique_relation_counts"]["verified_exact"],
                "conflict": qwen_17["unique_relation_counts"][
                    "catalog_conflict"
                ],
            },
        },
        "qwen3_1_7b_confirmation": compact_metrics(qwen_17),
        "qwen3_4b_cross_model_replication": compact_metrics(qwen_4),
        "title_counterfactual": {
            "row_count": title["row_count"],
            "focus_role_counts": title["focus_role_counts"],
            "choice_summary": title["choice_summary"],
            "catalog_sequence_summary": title["catalog_sequence_summary"],
            "technical_gate": title["technical_gate"],
        },
        "causal_trace": {
            "row_count": causal["row_count"],
            "example_count": causal["example_count"],
            "overall_earliest_sustained_residual_layer": causal["group_curves"][
                "all"
            ]["earliest_sustained_residual_layer"],
            "prior_masked_earliest_sustained_residual_layer": causal[
                "group_curves"
            ]["focus:sequence_exact_false_positive"][
                "earliest_sustained_residual_layer"
            ],
            "maximum_endpoint_error": causal["maximum_endpoint_error"],
            "maximum_contrast_reproduction_error": causal[
                "maximum_contrast_reproduction_error"
            ],
            "technical_gate": causal["technical_gate"],
        },
        "phase2_granite_confirmatory": {
            "playlist_count": phase2_receipt["final_counts"]["playlist_count"],
            "parsed_event_count": phase2_final["combined_row_count"],
            "unique_normalized_title_count": phase2_final[
                "combined_unique_normalized_title_count"
            ],
            "catalog_label_counts": phase2_final["combined_catalog_label_counts"],
            "strict_row_count": phase2_final["catalog_validation"][
                "strict_catalog_row_count"
            ],
            "unique_strict_conflict_title_clusters": phase2_final[
                "selected_conflict_cluster_count"
            ],
            "unique_strict_exact_title_clusters": phase2_final[
                "selected_exact_cluster_count"
            ],
            "minimum_conflicts_to_continue": phase2_final[
                "minimum_conflict_clusters_to_continue"
            ],
            "decision": phase2_final["decision"],
            "formal_mechanism_run_allowed": phase2_final[
                "formal_mechanism_run_allowed"
            ],
            "combined_evidence_raw_sha256": phase2_receipt["evidence_merge"][
                "combined_raw_sha256"
            ],
        },
        "phase3_qwen_relation_recoverability": {
            "generation_job_id": "6a55d4dfeffc02a91cbdc365",
            "recoverability_job_id": "6a55e748e4a4e82c0b592f5f",
            "recoverability_job_url": "https://huggingface.co/jobs/REDACTED/6a55e748e4a4e82c0b592f5f",
            "source_catalog_row_count": phase3_recoverability[
                "source_catalog_row_count"
            ],
            "selected_conflict_cluster_count": phase3_recoverability[
                "selected_conflict_cluster_count"
            ],
            "selected_exact_cluster_count": phase3_recoverability[
                "selected_exact_cluster_count"
            ],
            "canonical_recovery_fraction": phase3_recoverability[
                "canonical_recovery_fraction"
            ],
            "generated_exact_recovery_fraction": phase3_recoverability[
                "phase3_exact_recovery_fraction"
            ],
            "conflict_category_counts": phase3_recoverability[
                "conflict_category_counts"
            ],
            "recoverable_conflict_cluster_count": len(
                phase3_recoverability[
                    "recoverable_relation_conflict_record_ids"
                ]
            ),
            "minimum_recoverable_conflict_clusters": phase3_recoverability[
                "minimum_recoverable_conflict_clusters"
            ],
            "technical_gate": phase3_recoverability["technical_gate"],
            "assay_validity_gate": phase3_recoverability[
                "assay_validity_gate"
            ],
            "pilot_continuation_gate": phase3_recoverability[
                "pilot_continuation_gate"
            ],
            "decision": phase3_recoverability["pilot_decision"],
        },
        "hf_jobs": {
            role: {
                "job_id": job["job_id"],
                "url": job["url"],
                "status": job["status"],
                "embedded_script_sha256": job["embedded_script_sha256"],
                "terminal_log_sha256": job["terminal_log_sha256"],
            }
            for role, job in sorted(jobs_by_role.items())
        },
        "publication_file_count": len(publication_paths),
        "core_file_sha256": {
            path.as_posix(): sha256(PROJECT_ROOT / path)
            for path in publication_paths
        },
        "claim_boundaries": [
            "The pre-specified same-model warning rule was not confirmed.",
            "The protocol was frozen in-session before scoring but was not externally timestamped.",
            "Cross-model Qwen3-4B verification missed the same three unique conflict relations.",
            "Counterfactual title effects distinguish prior masking from wrong relation binding post hoc.",
            "Causal recovery follows the factual-title state and is not automatically factual correction.",
            "The study does not establish subjective awareness or a general hallucination detector.",
            "Phase 2 stopped at 7 unique strict conflicts versus a preregistered minimum of 30.",
            "Phase 2 H1 and H2 were not tested, so catalog-yield failure is not hypothesis rejection.",
            "Phase 3 stopped at 0 recoverable conflicts out of 11 catalog-conflict candidates versus a frozen minimum of 8.",
            "Positive reference-versus-emitted margins on four Phase 3 candidates did not establish candidate-free relation recovery.",
            "Neither internal diagnostics nor fluent reasons replace external catalog grounding.",
            "Raw third-party catalog response bodies are not redistributed in the public snapshot; their frozen hashes remain in a public receipt.",
        ],
    }


def main() -> int:
    output = PROJECT_ROOT / "runs" / "publication_manifest.json"
    output.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
