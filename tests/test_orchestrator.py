import json
from pathlib import Path

import pytest

from bio_integration_agent.contracts import (
    AlgorithmSpec,
    ClaimKind,
    Direction,
    EvidenceBundle,
    EvidenceClaim,
    EvidenceClass,
    ProjectConfig,
    SourceRecord,
    VerificationStatus,
)
from bio_integration_agent.evidence import EvidenceError
from bio_integration_agent.orchestrator import Orchestrator
from bio_integration_agent.state import ProjectStage


def make_project() -> ProjectConfig:
    return ProjectConfig(
        project_id="evidence_demo",
        direction=Direction.SPATIAL_DOMAIN_IDENTIFICATION,
        algorithm_set=(
            AlgorithmSpec(algorithm_id="method_a", canonical_name="Method A"),
            AlgorithmSpec(algorithm_id="method_b", canonical_name="Method B"),
        ),
        retrieval_scope={
            "independent_benchmarks": True,
            "biological_applications": True,
        },
    )


def verified_bundle(agent: Orchestrator) -> EvidenceBundle:
    sources = tuple(
        SourceRecord(
            source_id=f"paper_{algorithm_id}",
            source_type=EvidenceClass.ORIGINAL_PAPER,
            title=f"Primary paper for {algorithm_id}",
            url="not_specified",
            verification_status=VerificationStatus.VERIFIED,
        )
        for algorithm_id in ("method_a", "method_b")
    )
    claims = tuple(
        EvidenceClaim(
            claim_id=f"claim_{algorithm_id}",
            claim_text=f"Verified test-fixture claim for {algorithm_id}",
            algorithm_id=algorithm_id,
            source_id=f"paper_{algorithm_id}",
            source_type=EvidenceClass.ORIGINAL_PAPER,
            source_location="test fixture",
            evidence_excerpt_or_note="Synthetic unit-test metadata; not scientific evidence.",
            evidence_class=EvidenceClass.ORIGINAL_PAPER,
            claim_kind=ClaimKind.VERIFIED_FACT,
            verification_status=VerificationStatus.VERIFIED,
        )
        for algorithm_id in ("method_a", "method_b")
    )
    return EvidenceBundle(
        project_id=agent.config.project_id,
        configuration_hash=agent.configuration_hash,
        sources=sources,
        claims=claims,
    )


def write_bundle(path: Path, bundle: EvidenceBundle) -> None:
    path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")


def test_verified_evidence_advances_and_freezes_ledger(tmp_path: Path) -> None:
    agent = Orchestrator(make_project(), tmp_path)
    agent.validate()
    bundle_path = tmp_path / "bundle.json"
    write_bundle(bundle_path, verified_bundle(agent))

    audit = agent.import_evidence(bundle_path)

    assert audit.status == "PASS"
    assert audit.source_count == 2
    assert audit.claim_count == 2
    assert agent.state.stage == ProjectStage.RETRIEVE_EVIDENCE
    assert (tmp_path / "artifacts" / "evidence_ledger.json").is_file()
    assert agent.import_evidence(bundle_path) == audit


def test_empty_bundle_writes_blocking_audit_without_advancing(tmp_path: Path) -> None:
    agent = Orchestrator(make_project(), tmp_path)
    agent.validate()
    bundle_path = tmp_path / "empty_bundle.json"
    agent.write_evidence_template(bundle_path)

    with pytest.raises(EvidenceError, match="BLOCKED"):
        agent.import_evidence(bundle_path)

    assert agent.state.stage == ProjectStage.VALIDATE_INPUT
    assert not (tmp_path / "artifacts" / "evidence_ledger.json").exists()
    audit = json.loads(
        (tmp_path / "artifacts" / "evidence_audit.json").read_text(encoding="utf-8")
    )
    assert audit["status"] == "BLOCKED"
    assert {item["check"] for item in audit["findings"]} == {"algorithm_coverage"}


def test_bundle_for_changed_configuration_is_blocked(tmp_path: Path) -> None:
    agent = Orchestrator(make_project(), tmp_path)
    agent.validate()
    bundle = verified_bundle(agent).model_copy(update={"configuration_hash": "stale"})
    bundle_path = tmp_path / "stale_bundle.json"
    write_bundle(bundle_path, bundle)

    with pytest.raises(EvidenceError, match="BLOCKED"):
        agent.import_evidence(bundle_path)

    audit = json.loads(
        (tmp_path / "artifacts" / "evidence_audit.json").read_text(encoding="utf-8")
    )
    assert any(item["check"] == "configuration_identity" for item in audit["findings"])
