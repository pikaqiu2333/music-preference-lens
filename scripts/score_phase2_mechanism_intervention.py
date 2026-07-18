"""Join physically isolated Phase 2 outputs and apply frozen H1/H2 gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from phase2_mechanism_analysis import (
    classify_template,
    confirmatory_mechanism_label,
    h1_metrics,
    h2_interaction,
    score_candidate_free_condition,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def unique_by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output = {str(row["record_id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{label} contains duplicate record IDs")
    return output


def validate_provenance(
    diagnosis: dict[str, Any],
    correction: dict[str, Any],
    scoring_key: dict[str, Any],
    protocol: dict[str, Any],
    *,
    observed_protocol_hashes: dict[str, str] | None = None,
    observed_catalog_asset_hashes: dict[str, str] | None = None,
    require_execution_assets: bool = False,
) -> None:
    protocol_id = protocol["protocol_id"]
    for label, artifact in (
        ("diagnosis", diagnosis),
        ("correction", correction),
        ("scoring key", scoring_key),
    ):
        artifact_protocol_id = (
            artifact["summary"]["protocol_id"]
            if "summary" in artifact
            else artifact["protocol_id"]
        )
        if artifact_protocol_id != protocol_id:
            raise ValueError(f"{label} protocol ID mismatch")
    hashes = scoring_key["protocol_hashes"]
    if diagnosis["summary"]["protocol_hashes"] != hashes:
        raise ValueError("diagnosis protocol hashes differ from scoring key")
    if correction["summary"]["protocol_hashes"] != hashes:
        raise ValueError("correction protocol hashes differ from scoring key")
    if observed_protocol_hashes is not None and hashes != observed_protocol_hashes:
        raise ValueError("local protocol bytes differ from frozen scoring assets")
    expected_model = protocol["models"]["confirmatory"]
    if scoring_key.get("model_id") != expected_model["model_id"]:
        raise ValueError("scoring-key model ID mismatch")
    if scoring_key.get("model_revision") != expected_model["revision"]:
        raise ValueError("scoring-key model revision mismatch")
    for label, artifact in (("diagnosis", diagnosis), ("correction", correction)):
        summary = artifact["summary"]
        if summary.get("model_id") != expected_model["model_id"]:
            raise ValueError(f"{label} model ID mismatch")
        if summary.get("model_revision") != expected_model["revision"]:
            raise ValueError(f"{label} model revision mismatch")
    expected_catalog = scoring_key.get("catalog_asset_hashes", {})
    for label, artifact in (("diagnosis", diagnosis), ("correction", correction)):
        if artifact["summary"].get("catalog_asset_hashes") != expected_catalog:
            raise ValueError(f"{label} catalog asset hashes mismatch")
    if observed_catalog_asset_hashes is not None:
        if expected_catalog != observed_catalog_asset_hashes:
            raise ValueError("local catalog assets differ from scoring key")
    execution = scoring_key.get("execution_assets")
    if require_execution_assets and not execution:
        raise ValueError("scoring key has no bound execution assets")
    if execution:
        for label, artifact in (("diagnosis", diagnosis), ("correction", correction)):
            expected = execution[label]
            if not expected.get("canonical_bundle_sha256"):
                raise ValueError(f"{label} expected bundle hash is missing")
            if require_execution_assets and not expected.get("submitted_script_sha256"):
                raise ValueError(f"{label} expected runner hash is missing")
            if (
                artifact["summary"].get("canonical_bundle_sha256")
                != expected["canonical_bundle_sha256"]
            ):
                raise ValueError(f"{label} canonical bundle hash mismatch")
            if expected.get("submitted_script_sha256") and (
                artifact["summary"].get("submitted_script_sha256")
                != expected["submitted_script_sha256"]
            ):
                raise ValueError(f"{label} submitted runner hash mismatch")
    if not diagnosis["summary"].get("technical_gate"):
        raise ValueError("diagnosis Job technical gate failed")
    if not correction["summary"].get("technical_gate"):
        raise ValueError("correction Job technical gate failed")


def score_phase2(
    diagnosis: dict[str, Any],
    correction: dict[str, Any],
    scoring_key: dict[str, Any],
    protocol: dict[str, Any],
    *,
    observed_protocol_hashes: dict[str, str] | None = None,
    observed_catalog_asset_hashes: dict[str, str] | None = None,
    require_execution_assets: bool = False,
) -> dict[str, Any]:
    validate_provenance(
        diagnosis,
        correction,
        scoring_key,
        protocol,
        observed_protocol_hashes=observed_protocol_hashes,
        observed_catalog_asset_hashes=observed_catalog_asset_hashes,
        require_execution_assets=require_execution_assets,
    )
    diagnosis_by_id = unique_by_id(diagnosis["records"], "diagnosis")
    correction_by_id = unique_by_id(correction["records"], "correction")
    key_by_id = unique_by_id(scoring_key["records"], "scoring key")
    conflict_keys = {
        record_id: row
        for record_id, row in key_by_id.items()
        if row["record_type"] == "strict_conflict"
    }
    if set(diagnosis_by_id) != set(conflict_keys):
        raise ValueError("diagnosis and strict-conflict scoring IDs differ")
    if set(correction_by_id) != set(key_by_id):
        raise ValueError("correction and scoring-key IDs differ")

    diagnosis_spec = protocol["mechanism_diagnosis"]
    mechanism_rows = []
    technical_failure_count = 0
    for record_id, row in diagnosis_by_id.items():
        if len(row["template_results"]) != 2:
            raise ValueError("diagnosis requires exactly two template results")
        template_labels = []
        template_details = []
        technical_failure = False
        for template in row["template_results"]:
            failure = template.get("technical_failure")
            medians = {
                int(layer): float(value)
                for layer, value in template.get("layer_median_shifts", {}).items()
            }
            if failure or set(medians) != set(diagnosis_spec["layers"]):
                technical_failure = True
                label = "indeterminate"
            else:
                label = classify_template(
                    medians,
                    minimum_absolute_shift=float(
                        diagnosis_spec["minimum_absolute_patch_relation_shift_nats"]
                    ),
                    minimum_layers_same_direction=int(
                        diagnosis_spec["minimum_layers_same_direction"]
                    ),
                )
            template_labels.append(label)
            template_details.append(
                {
                    "label": label,
                    "layer_median_shifts": {str(key): value for key, value in medians.items()},
                    "technical_failure": failure,
                }
            )
        if technical_failure:
            technical_failure_count += 1
        final_label = confirmatory_mechanism_label(template_labels)
        mechanism_rows.append(
            {
                "record_id": record_id,
                "normalized_title": conflict_keys[record_id]["normalized_title"],
                "template_labels": template_labels,
                "template_details": template_details,
                "technical_failure": technical_failure,
                "mechanism_label": final_label,
                "manipulation_check": row.get("manipulation_check"),
            }
        )

    h1_gate = protocol["hypothesis_gates"]
    h1 = h1_metrics(
        mechanism_rows,
        minimum_raw_agreement=float(h1_gate["h1_minimum_raw_three_class_agreement"]),
        minimum_kappa=float(h1_gate["h1_minimum_cohens_kappa"]),
        minimum_classifiable_coverage=float(
            h1_gate["h1_minimum_same_non_indeterminate_coverage"]
        ),
    )
    technical_failure_rate = technical_failure_count / len(mechanism_rows)
    technical_failure_rate_gate = technical_failure_rate <= float(
        diagnosis_spec["patch_specification"][
            "maximum_selected_cluster_technical_failure_rate"
        ]
    )
    h1["mechanism_technical_failure_count"] = technical_failure_count
    h1["mechanism_technical_failure_rate"] = technical_failure_rate
    h1["mechanism_technical_failure_rate_gate"] = technical_failure_rate_gate
    h1["confirmatory_passes"] = h1["passes"] and technical_failure_rate_gate

    mechanism_by_id = {row["record_id"]: row for row in mechanism_rows}
    correction_rows = []
    for record_id, key in key_by_id.items():
        raw = correction_by_id[record_id]
        condition_scores = {}
        for condition in (
            "naive_candidate_free_self_check",
            "anti_prior_candidate_free_recall",
        ):
            responses = [row["response"] for row in raw["conditions"][condition]]
            condition_scores[condition] = score_candidate_free_condition(
                responses, key["reference_artist"]
            )
        mechanism = mechanism_by_id.get(record_id)
        correction_rows.append(
            {
                "record_id": record_id,
                "record_type": key["record_type"],
                "normalized_title": key["normalized_title"],
                "mechanism_label": (
                    mechanism["mechanism_label"] if mechanism is not None else None
                ),
                "naive": condition_scores["naive_candidate_free_self_check"],
                "anti_prior": condition_scores["anti_prior_candidate_free_recall"],
                "catalog_safety_action": key["catalog_safety_action"],
            }
        )

    conflict_corrections = [
        row for row in correction_rows if row["record_type"] == "strict_conflict"
    ]
    h2_rows = [
        {
            "title": row["normalized_title"],
            "normalized_title": row["normalized_title"],
            "mechanism_label": row["mechanism_label"],
            "naive_accuracy": row["naive"]["accuracy"],
            "anti_prior_accuracy": row["anti_prior"]["accuracy"],
        }
        for row in conflict_corrections
    ]
    if h1["confirmatory_passes"]:
        h2 = h2_interaction(
            h2_rows,
            bootstrap_samples=int(h1_gate["bootstrap_samples"]),
            bootstrap_seed=int(h1_gate["bootstrap_seed"]),
            minimum_clusters_per_class=int(
                h1_gate["h2_minimum_normalized_title_clusters_per_class"]
            ),
            minimum_interaction=float(
                h1_gate["h2_minimum_interaction_percentage_points"]
            )
            / 100.0,
        )
        h2["tested"] = bool(h2.get("estimable"))
    else:
        h2 = {
            "tested": False,
            "estimable": False,
            "passes": False,
            "reason": "H1 or mechanism technical gate failed",
        }

    exact_controls = [
        row
        for row in correction_rows
        if row["record_type"] == "strict_exact_control"
    ]
    descriptive = {
        "mechanism_label_counts": dict(
            Counter(row["mechanism_label"] for row in mechanism_rows)
        ),
        "conflict_naive_mean_accuracy": (
            sum(row["naive"]["accuracy"] for row in conflict_corrections)
            / len(conflict_corrections)
            if conflict_corrections
            else None
        ),
        "conflict_anti_prior_mean_accuracy": (
            sum(row["anti_prior"]["accuracy"] for row in conflict_corrections)
            / len(conflict_corrections)
            if conflict_corrections
            else None
        ),
        "exact_control_naive_mean_accuracy": (
            sum(row["naive"]["accuracy"] for row in exact_controls)
            / len(exact_controls)
            if exact_controls
            else None
        ),
        "exact_control_anti_prior_mean_accuracy": (
            sum(row["anti_prior"]["accuracy"] for row in exact_controls)
            / len(exact_controls)
            if exact_controls
            else None
        ),
        "catalog_safety_reference_available_rate": sum(
            row["catalog_safety_action"] == "return_reference" for row in correction_rows
        )
        / len(correction_rows),
    }
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_hashes": scoring_key["protocol_hashes"],
        "H1": h1,
        "H2": h2,
        "decision": (
            "CONFIRMATORY_PASS"
            if h1["confirmatory_passes"] and h2.get("passes")
            else "STOP_AND_REPORT_BOUNDARY_OR_NEGATIVE_RESULT"
        ),
        "mechanism_rows": mechanism_rows,
        "correction_rows": correction_rows,
        "descriptive": descriptive,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis-artifact", type=Path, required=True)
    parser.add_argument("--correction-artifact", type=Path, required=True)
    parser.add_argument("--scoring-key", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-doc", type=Path, required=True)
    parser.add_argument("--catalog-rows", type=Path, required=True)
    parser.add_argument("--catalog-evidence-archive", type=Path, required=True)
    parser.add_argument("--catalog-verifier-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_bytes = args.protocol.read_bytes()
    protocol_doc_bytes = args.protocol_doc.read_bytes()
    observed_protocol_hashes = {
        "json_sha256": sha256_bytes(protocol_bytes),
        "markdown_sha256": sha256_bytes(protocol_doc_bytes),
    }
    observed_catalog_asset_hashes = {
        "catalog_rows_sha256": sha256_bytes(args.catalog_rows.read_bytes()),
        "catalog_evidence_sha256": sha256_bytes(
            args.catalog_evidence_archive.read_bytes()
        ),
        "catalog_verifier_script_sha256": sha256_bytes(
            args.catalog_verifier_script.read_bytes()
        ),
    }
    result = score_phase2(
        load_json(args.diagnosis_artifact),
        load_json(args.correction_artifact),
        load_json(args.scoring_key),
        json.loads(protocol_bytes.decode("utf-8")),
        observed_protocol_hashes=observed_protocol_hashes,
        observed_catalog_asset_hashes=observed_catalog_asset_hashes,
        require_execution_assets=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE2_FINAL_SUMMARY_JSON=" + json.dumps({
        "H1": result["H1"],
        "H2": result["H2"],
        "decision": result["decision"],
        "descriptive": result["descriptive"],
    }, ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
