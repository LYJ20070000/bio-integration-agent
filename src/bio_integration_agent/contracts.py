from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from math import isfinite
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Direction(StrEnum):
    SPATIAL_DOMAIN_IDENTIFICATION = "spatial_domain_identification"
    SINGLE_CELL_INTEGRATION = "single_cell_integration"
    HISTOLOGY_TO_SPATIAL_EXPRESSION = "histology_to_spatial_expression"


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    INACCESSIBLE = "inaccessible"
    PENDING = "pending"


class EvidenceClass(StrEnum):
    ORIGINAL_PAPER = "original_paper"
    OFFICIAL_DOCUMENTATION = "official_documentation"
    OFFICIAL_REPOSITORY = "official_repository"
    INDEPENDENT_BENCHMARK = "independent_benchmark"
    BIOLOGICAL_APPLICATION = "biological_application"
    BACKGROUND_REFERENCE = "background_reference"


class ClaimKind(StrEnum):
    VERIFIED_FACT = "verified_fact"
    SOURCE_AUTHOR_CLAIM = "source_author_claim"
    INDEPENDENT_EVIDENCE = "independent_evidence"
    AGENT_INFERENCE = "agent_inference"
    UNVERIFIED = "unverified"


class ResultStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


class AlgorithmSpec(StrictModel):
    algorithm_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*$")]
    canonical_name: str
    aliases: tuple[str, ...] = ()
    original_paper_doi: str = "not_specified"
    original_paper_url: HttpUrl | Literal["not_specified"] = "not_specified"
    version: str = "not_specified"
    related_versions: tuple[str, ...] = ()
    documentation_url: HttpUrl | Literal["not_specified"] = "not_specified"
    repository_url: HttpUrl | Literal["not_specified"] = "not_specified"
    notes: str = "not_specified"


class ProjectConfig(StrictModel):
    project_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*$")]
    direction: Direction
    algorithm_set: tuple[AlgorithmSpec, ...]
    retrieval_scope: dict[str, Any] = Field(default_factory=dict)
    available_materials: dict[str, Any] = Field(default_factory=dict)
    evaluation_preferences: dict[str, Any] = Field(default_factory=dict)
    downstream_analysis_requested: bool = False

    @model_validator(mode="after")
    def validate_closed_set(self) -> "ProjectConfig":
        ids = [item.algorithm_id for item in self.algorithm_set]
        if len(ids) < 2:
            raise ValueError("algorithm_set must contain at least two algorithms")
        if len(ids) != len(set(ids)):
            raise ValueError("algorithm_set contains duplicate algorithm_id values")
        return self

    @property
    def algorithm_ids(self) -> frozenset[str]:
        return frozenset(item.algorithm_id for item in self.algorithm_set)

    def missing_fields(self) -> dict[str, list[str]]:
        missing: dict[str, list[str]] = {}
        optional = (
            "original_paper_doi",
            "original_paper_url",
            "version",
            "documentation_url",
            "repository_url",
        )
        for algorithm in self.algorithm_set:
            fields = [name for name in optional if getattr(algorithm, name) == "not_specified"]
            if fields:
                missing[algorithm.algorithm_id] = fields
        return missing


class SourceRecord(StrictModel):
    source_id: str
    source_type: EvidenceClass
    title: str
    url: HttpUrl | Literal["inaccessible", "not_specified"]
    doi: str = "not_specified"
    retrieval_date: date = Field(default_factory=date.today)
    verification_status: VerificationStatus = VerificationStatus.PENDING


class EvidenceClaim(StrictModel):
    claim_id: str
    claim_text: str
    algorithm_id: str
    source_id: str
    source_type: EvidenceClass
    source_location: str
    evidence_excerpt_or_note: str
    evidence_class: EvidenceClass
    claim_kind: ClaimKind
    verification_status: VerificationStatus
    retrieval_date: date = Field(default_factory=date.today)
    supporting_claim_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_provenance_fields(self) -> "EvidenceClaim":
        if self.source_type != self.evidence_class:
            raise ValueError("source_type and evidence_class must match")
        if self.claim_kind == ClaimKind.AGENT_INFERENCE and not self.supporting_claim_ids:
            raise ValueError("agent inference must reference supporting_claim_ids")
        if (
            self.claim_kind == ClaimKind.AGENT_INFERENCE
            and self.verification_status == VerificationStatus.VERIFIED
        ):
            raise ValueError("agent inference cannot be labeled as a verified source fact")
        if self.claim_kind == ClaimKind.UNVERIFIED and self.verification_status == VerificationStatus.VERIFIED:
            raise ValueError("an unverified claim cannot have verified status")
        return self


class EvidenceBundle(StrictModel):
    project_id: str
    configuration_hash: str
    sources: tuple[SourceRecord, ...] = ()
    claims: tuple[EvidenceClaim, ...] = ()

    @model_validator(mode="after")
    def require_unique_identifiers(self) -> "EvidenceBundle":
        for name, identifiers in (
            ("source_id", [item.source_id for item in self.sources]),
            ("claim_id", [item.claim_id for item in self.claims]),
        ):
            duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
            if duplicates:
                raise ValueError(f"duplicate {name} values: {duplicates}")
        return self


class EvidenceAuditFinding(StrictModel):
    check: str
    severity: Literal["blocking", "warning"]
    detail: str


class EvidenceAuditReport(StrictModel):
    status: Literal["PASS", "BLOCKED"]
    project_id: str
    configuration_hash: str
    bundle_hash: str
    source_count: int
    claim_count: int
    verified_claim_count: int
    findings: tuple[EvidenceAuditFinding, ...] = ()
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def status_matches_findings(self) -> "EvidenceAuditReport":
        has_blocker = any(item.severity == "blocking" for item in self.findings)
        expected = "BLOCKED" if has_blocker else "PASS"
        if self.status != expected:
            raise ValueError(f"evidence status must be {expected} for the supplied findings")
        return self


class IntegrationComponent(StrictModel):
    component_id: str
    source_algorithm_id: str
    source_component: str
    adapted_component: str
    target_weakness: str
    evidence_claim_ids: tuple[str, ...]
    new_assumptions: tuple[str, ...] = ()
    ablation_flag: str


class IntegrationDesign(StrictModel):
    """Auditable declaration of how closed-set components form one candidate."""

    method_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*$")]
    components: tuple[IntegrationComponent, ...]
    composition_description: str
    objective_metric: str
    higher_is_better: bool = True
    minimum_improvement: Annotated[float, Field(ge=0.0)] = 0.0
    minimum_complete_comparisons: Annotated[int, Field(ge=1)] = 1
    result_scope: Literal["DEMO_ONLY", "SCIENTIFIC"]

    @model_validator(mode="after")
    def validate_components(self) -> "IntegrationDesign":
        if not self.components:
            raise ValueError("integration design requires at least one component")
        component_ids = [item.component_id for item in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("integration design contains duplicate component_id values")
        ablation_flags = [item.ablation_flag for item in self.components]
        if len(ablation_flags) != len(set(ablation_flags)):
            raise ValueError("each integration component requires a unique ablation_flag")
        if self.result_scope == "SCIENTIFIC" and any(
            not item.evidence_claim_ids for item in self.components
        ):
            raise ValueError("scientific integration components require evidence_claim_ids")
        return self

    def validate_for(self, project: ProjectConfig) -> None:
        supplied = project.algorithm_ids
        represented = frozenset(item.source_algorithm_id for item in self.components)
        if represented != supplied:
            missing = sorted(supplied - represented)
            extra = sorted(represented - supplied)
            raise ValueError(
                "integration design must use exactly the closed algorithm set; "
                f"missing={missing}, extra={extra}"
            )


class IntegrationAssessment(StrictModel):
    candidate_method_id: str
    status: Literal[
        "DEMO_ONLY", "IMPROVEMENT_SUPPORTED", "NO_IMPROVEMENT", "INCONCLUSIVE"
    ]
    result_scope: Literal["DEMO_ONLY", "SCIENTIFIC"]
    metric: str
    higher_is_better: bool
    minimum_improvement: float
    complete_comparisons: int
    planned_comparisons: int
    mean_margin_over_best_baseline: float | None = None
    worst_margin_over_best_baseline: float | None = None
    reason: str


class LeakageFinding(StrictModel):
    check: str
    severity: Literal["blocking", "warning"]
    detail: str


class LeakageReport(StrictModel):
    status: Literal["PASS", "BLOCKED"]
    findings: tuple[LeakageFinding, ...] = ()
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def status_matches_findings(self) -> "LeakageReport":
        has_blocker = any(item.severity == "blocking" for item in self.findings)
        expected = "BLOCKED" if has_blocker else "PASS"
        if self.status != expected:
            raise ValueError(f"leakage status must be {expected} for the supplied findings")
        return self


class EvaluationPlan(StrictModel):
    """Frozen comparison policy created before any result is calculated."""

    baseline_method_ids: tuple[str, ...]
    integrated_method_ids: tuple[str, ...] = ()
    datasets: tuple[str, ...]
    splits: tuple[str, ...]
    metrics: tuple[str, ...]
    seeds: tuple[int, ...]
    split_unit: str
    preprocessing_policy: str
    hyperparameter_tuning_policy: str
    result_scope: Literal["DEMO_ONLY", "SCIENTIFIC"]

    @model_validator(mode="after")
    def require_complete_nonduplicate_fields(self) -> "EvaluationPlan":
        required = {
            "baseline_method_ids": self.baseline_method_ids,
            "datasets": self.datasets,
            "splits": self.splits,
            "metrics": self.metrics,
            "seeds": self.seeds,
        }
        for name, values in required.items():
            if not values:
                raise ValueError(f"{name} cannot be empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} contains duplicates")
        overlap = set(self.baseline_method_ids) & set(self.integrated_method_ids)
        if overlap:
            raise ValueError(f"baseline and integrated method ids overlap: {sorted(overlap)}")
        return self

    def validate_for(self, project: ProjectConfig) -> None:
        supplied = project.algorithm_ids
        planned = frozenset(self.baseline_method_ids)
        if planned != supplied:
            missing = sorted(supplied - planned)
            extra = sorted(planned - supplied)
            raise ValueError(
                f"evaluation plan must include exactly the closed-set baselines; "
                f"missing={missing}, extra={extra}"
            )


class ResultRecord(StrictModel):
    method: str
    dataset: str
    split: str
    seed: int
    configuration_hash: str
    metric: str
    value: float | None = None
    status: ResultStatus = ResultStatus.SUCCESS
    failure_reason: str = "not_applicable"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    code_commit: str = "not_specified"
    result_scope: Literal["DEMO_ONLY", "SCIENTIFIC"] = "DEMO_ONLY"
    implementation_kind: Literal[
        "official", "mechanism_demonstration", "custom", "unverified"
    ] = "unverified"
    evaluation_plan_hash: str = "not_specified"
    leakage_report_hash: str = "not_specified"

    @model_validator(mode="after")
    def validate_outcome(self) -> "ResultRecord":
        if self.status == ResultStatus.SUCCESS:
            if self.value is None or not isfinite(self.value):
                raise ValueError("successful results require a finite value")
            if self.failure_reason != "not_applicable":
                raise ValueError("successful results cannot have a failure_reason")
        elif self.status == ResultStatus.FAILED:
            if self.value is not None:
                raise ValueError("failed results cannot contain a metric value")
            if self.failure_reason in {"", "not_applicable", "not_specified"}:
                raise ValueError("failed results require a failure_reason")
        elif self.value is not None:
            raise ValueError("pending results cannot contain a metric value")
        if self.result_scope == "SCIENTIFIC":
            if self.code_commit == "not_specified":
                raise ValueError("scientific results require code_commit provenance")
            if self.implementation_kind == "unverified":
                raise ValueError("scientific results require verified implementation_kind")
        if self.implementation_kind == "mechanism_demonstration" and self.result_scope != "DEMO_ONLY":
            raise ValueError("mechanism demonstrations must remain DEMO_ONLY")
        return self
