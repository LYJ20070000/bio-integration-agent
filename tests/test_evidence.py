import pytest

from bio_integration_agent.contracts import (
    AlgorithmSpec,
    Direction,
    EvidenceClaim,
    EvidenceClass,
    ProjectConfig,
    SourceRecord,
    VerificationStatus,
    ClaimKind,
)
from bio_integration_agent.evidence import ClaimStore, EvidenceError


def project() -> ProjectConfig:
    return ProjectConfig(
        project_id="demo",
        direction=Direction.SPATIAL_DOMAIN_IDENTIFICATION,
        algorithm_set=(
            AlgorithmSpec(algorithm_id="method_a", canonical_name="Method A"),
            AlgorithmSpec(algorithm_id="method_b", canonical_name="Method B"),
        ),
    )


def test_rejects_claim_for_algorithm_outside_closed_set() -> None:
    store = ClaimStore(project())
    store.add_source(
        SourceRecord(
            source_id="paper_1",
            source_type=EvidenceClass.ORIGINAL_PAPER,
            title="Paper",
            url="not_specified",
        )
    )
    claim = EvidenceClaim(
        claim_id="claim_1",
        claim_text="Unsupported external method claim",
        algorithm_id="method_c",
        source_id="paper_1",
        source_type=EvidenceClass.ORIGINAL_PAPER,
        source_location="page 1",
        evidence_excerpt_or_note="test",
        evidence_class=EvidenceClass.ORIGINAL_PAPER,
        claim_kind=ClaimKind.UNVERIFIED,
        verification_status=VerificationStatus.UNVERIFIED,
    )
    with pytest.raises(EvidenceError, match="outside project.algorithm_set"):
        store.add_claim(claim)


def test_agent_inference_requires_supporting_claims() -> None:
    with pytest.raises(ValueError, match="supporting_claim_ids"):
        EvidenceClaim(
            claim_id="inference_1",
            claim_text="An inference",
            algorithm_id="method_a",
            source_id="paper_1",
            source_type=EvidenceClass.ORIGINAL_PAPER,
            source_location="synthesis",
            evidence_excerpt_or_note="reasoning",
            evidence_class=EvidenceClass.ORIGINAL_PAPER,
            claim_kind=ClaimKind.AGENT_INFERENCE,
            verification_status=VerificationStatus.UNVERIFIED,
        )


def test_verified_claim_requires_matching_verified_source() -> None:
    store = ClaimStore(project())
    store.add_source(
        SourceRecord(
            source_id="paper_1",
            source_type=EvidenceClass.ORIGINAL_PAPER,
            title="Paper",
            url="not_specified",
            verification_status=VerificationStatus.PENDING,
        )
    )
    claim = EvidenceClaim(
        claim_id="claim_1",
        claim_text="A purported verified fact",
        algorithm_id="method_a",
        source_id="paper_1",
        source_type=EvidenceClass.ORIGINAL_PAPER,
        source_location="page 1",
        evidence_excerpt_or_note="test fixture",
        evidence_class=EvidenceClass.ORIGINAL_PAPER,
        claim_kind=ClaimKind.VERIFIED_FACT,
        verification_status=VerificationStatus.VERIFIED,
    )
    with pytest.raises(EvidenceError, match="verified source"):
        store.add_claim(claim)


def test_agent_inference_cannot_be_labeled_verified() -> None:
    with pytest.raises(ValueError, match="inference cannot"):
        EvidenceClaim(
            claim_id="inference_1",
            claim_text="A synthesis",
            algorithm_id="method_a",
            source_id="paper_1",
            source_type=EvidenceClass.ORIGINAL_PAPER,
            source_location="synthesis",
            evidence_excerpt_or_note="test fixture",
            evidence_class=EvidenceClass.ORIGINAL_PAPER,
            claim_kind=ClaimKind.AGENT_INFERENCE,
            verification_status=VerificationStatus.VERIFIED,
            supporting_claim_ids=("claim_1",),
        )
