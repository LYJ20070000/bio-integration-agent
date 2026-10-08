from __future__ import annotations

from .contracts import (
    ClaimKind,
    EvidenceAuditFinding,
    EvidenceAuditReport,
    EvidenceBundle,
    EvidenceClaim,
    EvidenceClass,
    ProjectConfig,
    SourceRecord,
    VerificationStatus,
)
from .evaluation import stable_hash


class EvidenceError(ValueError):
    pass


class ClaimStore:
    def __init__(self, project: ProjectConfig) -> None:
        self._allowed_algorithms = project.algorithm_ids
        self._sources: dict[str, SourceRecord] = {}
        self._claims: dict[str, EvidenceClaim] = {}

    def add_source(self, source: SourceRecord) -> None:
        if source.source_id in self._sources:
            raise EvidenceError(f"duplicate source_id: {source.source_id}")
        self._sources[source.source_id] = source

    def add_claim(self, claim: EvidenceClaim) -> None:
        if claim.algorithm_id not in self._allowed_algorithms:
            raise EvidenceError(
                f"algorithm '{claim.algorithm_id}' is outside project.algorithm_set"
            )
        if claim.source_id not in self._sources:
            raise EvidenceError(f"unknown source_id: {claim.source_id}")
        source = self._sources[claim.source_id]
        if claim.source_type != source.source_type:
            raise EvidenceError(
                f"claim source_type '{claim.source_type}' does not match registered source "
                f"type '{source.source_type}'"
            )
        if (
            claim.verification_status == VerificationStatus.VERIFIED
            and source.verification_status != VerificationStatus.VERIFIED
        ):
            raise EvidenceError("verified claims require a verified source")
        if claim.claim_id in self._claims:
            raise EvidenceError(f"duplicate claim_id: {claim.claim_id}")
        missing = set(claim.supporting_claim_ids) - self._claims.keys()
        if missing:
            raise EvidenceError(f"unknown supporting claims: {sorted(missing)}")
        self._claims[claim.claim_id] = claim

    def get_claim(self, claim_id: str) -> EvidenceClaim:
        try:
            return self._claims[claim_id]
        except KeyError as exc:
            raise EvidenceError(f"unknown claim_id: {claim_id}") from exc

    def export(self) -> dict[str, list[dict[str, object]]]:
        return {
            "sources": [item.model_dump(mode="json") for item in self._sources.values()],
            "claims": [item.model_dump(mode="json") for item in self._claims.values()],
        }


def audit_evidence_bundle(
    project: ProjectConfig,
    configuration_hash: str,
    bundle: EvidenceBundle,
) -> EvidenceAuditReport:
    """Validate provenance and minimum evidence needed to begin method profiling."""

    findings: list[EvidenceAuditFinding] = []
    if bundle.project_id != project.project_id:
        findings.append(
            EvidenceAuditFinding(
                check="project_identity",
                severity="blocking",
                detail=(
                    f"bundle project_id '{bundle.project_id}' does not match "
                    f"'{project.project_id}'"
                ),
            )
        )
    if bundle.configuration_hash != configuration_hash:
        findings.append(
            EvidenceAuditFinding(
                check="configuration_identity",
                severity="blocking",
                detail="bundle configuration_hash does not match the immutable project configuration",
            )
        )

    store = ClaimStore(project)
    for source in bundle.sources:
        try:
            store.add_source(source)
        except EvidenceError as exc:
            findings.append(
                EvidenceAuditFinding(
                    check="source_provenance",
                    severity="blocking",
                    detail=str(exc),
                )
            )

    accepted_claims: list[EvidenceClaim] = []
    for claim in bundle.claims:
        try:
            store.add_claim(claim)
            accepted_claims.append(claim)
        except EvidenceError as exc:
            findings.append(
                EvidenceAuditFinding(
                    check="claim_provenance",
                    severity="blocking",
                    detail=f"{claim.claim_id}: {exc}",
                )
            )

    primary_classes = {
        EvidenceClass.ORIGINAL_PAPER,
        EvidenceClass.OFFICIAL_DOCUMENTATION,
        EvidenceClass.OFFICIAL_REPOSITORY,
    }
    for algorithm_id in sorted(project.algorithm_ids):
        algorithm_claims = [
            claim for claim in accepted_claims if claim.algorithm_id == algorithm_id
        ]
        if not algorithm_claims:
            findings.append(
                EvidenceAuditFinding(
                    check="algorithm_coverage",
                    severity="blocking",
                    detail=f"{algorithm_id}: no valid claims supplied",
                )
            )
            continue
        verified_primary = [
            claim
            for claim in algorithm_claims
            if claim.evidence_class in primary_classes
            and claim.verification_status == VerificationStatus.VERIFIED
            and claim.claim_kind != ClaimKind.AGENT_INFERENCE
        ]
        if not verified_primary:
            findings.append(
                EvidenceAuditFinding(
                    check="verified_primary_evidence",
                    severity="blocking",
                    detail=f"{algorithm_id}: no verified primary-source claim supplied",
                )
            )
        optional_scopes = (
            ("independent_benchmarks", EvidenceClass.INDEPENDENT_BENCHMARK),
            ("biological_applications", EvidenceClass.BIOLOGICAL_APPLICATION),
        )
        for scope_name, evidence_class in optional_scopes:
            if project.retrieval_scope.get(scope_name) and not any(
                claim.evidence_class == evidence_class
                and claim.verification_status == VerificationStatus.VERIFIED
                for claim in algorithm_claims
            ):
                findings.append(
                    EvidenceAuditFinding(
                        check=scope_name,
                        severity="warning",
                        detail=(
                            f"{algorithm_id}: bundle contains no verified {scope_name} claim; "
                            "absence from this bundle is not evidence that none exists"
                        ),
                    )
                )

    blocking = any(item.severity == "blocking" for item in findings)
    return EvidenceAuditReport(
        status="BLOCKED" if blocking else "PASS",
        project_id=project.project_id,
        configuration_hash=configuration_hash,
        bundle_hash=stable_hash(bundle.model_dump(mode="json")),
        source_count=len(bundle.sources),
        claim_count=len(bundle.claims),
        verified_claim_count=sum(
            claim.verification_status == VerificationStatus.VERIFIED
            for claim in accepted_claims
        ),
        findings=tuple(findings),
    )
