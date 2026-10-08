from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bio_integration_agent.contracts import (
    AlgorithmSpec,
    ClaimKind,
    Direction,
    EvaluationPlan,
    EvidenceClaim,
    EvidenceClass,
    IntegrationComponent,
    IntegrationDesign,
    ProjectConfig,
    SourceRecord,
    VerificationStatus,
)
from bio_integration_agent.evaluation import audit_leakage
from bio_integration_agent.evidence import ClaimStore
from bio_integration_agent.integration import EvaluationCase, IntegrationEvaluator
from bio_integration_agent.methods import FloatArray, SpatialMethod


class ConstantMethod(SpatialMethod):
    def __init__(self, method_id: str, value: float, *, fail: bool = False) -> None:
        self.method_id = method_id
        self.value = value
        self.fail = fail
        self.calls = 0

    def fit_transform(
        self,
        expression: FloatArray,
        coordinates: FloatArray,
        image_features: FloatArray | None = None,
    ) -> FloatArray:
        self.calls += 1
        if self.fail:
            raise RuntimeError("deliberate integration test failure")
        return np.full_like(expression, self.value, dtype=float)


def project() -> ProjectConfig:
    return ProjectConfig(
        project_id="integration_test",
        direction=Direction.SPATIAL_DOMAIN_IDENTIFICATION,
        algorithm_set=(
            AlgorithmSpec(algorithm_id="method_a", canonical_name="Method A"),
            AlgorithmSpec(algorithm_id="method_b", canonical_name="Method B"),
        ),
    )


def claims(project_config: ProjectConfig) -> ClaimStore:
    store = ClaimStore(project_config)
    for algorithm_id in sorted(project_config.algorithm_ids):
        source_id = f"source_{algorithm_id}"
        store.add_source(
            SourceRecord(
                source_id=source_id,
                source_type=EvidenceClass.ORIGINAL_PAPER,
                title=f"Test source for {algorithm_id}",
                url="not_specified",
                verification_status=VerificationStatus.VERIFIED,
            )
        )
        store.add_claim(
            EvidenceClaim(
                claim_id=f"claim_{algorithm_id}",
                claim_text=f"Verified test-fixture mechanism for {algorithm_id}",
                algorithm_id=algorithm_id,
                source_id=source_id,
                source_type=EvidenceClass.ORIGINAL_PAPER,
                source_location="unit test fixture",
                evidence_excerpt_or_note="Not scientific evidence; validates software gates only.",
                evidence_class=EvidenceClass.ORIGINAL_PAPER,
                claim_kind=ClaimKind.VERIFIED_FACT,
                verification_status=VerificationStatus.VERIFIED,
            )
        )
    return store


def design() -> IntegrationDesign:
    return IntegrationDesign(
        method_id="candidate",
        components=(
            IntegrationComponent(
                component_id="component_a",
                source_algorithm_id="method_a",
                source_component="fixture A",
                adapted_component="adapter A",
                target_weakness="fixture weakness",
                evidence_claim_ids=("claim_method_a",),
                ablation_flag="without_a",
            ),
            IntegrationComponent(
                component_id="component_b",
                source_algorithm_id="method_b",
                source_component="fixture B",
                adapted_component="adapter B",
                target_weakness="fixture weakness",
                evidence_claim_ids=("claim_method_b",),
                ablation_flag="without_b",
            ),
        ),
        composition_description="Unit-test composition",
        objective_metric="fixture_score",
        minimum_improvement=0.1,
        result_scope="SCIENTIFIC",
    )


def plan() -> EvaluationPlan:
    return EvaluationPlan(
        baseline_method_ids=("method_a", "method_b"),
        integrated_method_ids=("candidate", "without_a", "without_b"),
        datasets=("fixture",),
        splits=("held_out",),
        metrics=("fixture_score",),
        seeds=(1,),
        split_unit="patient",
        preprocessing_policy="train_only",
        hyperparameter_tuning_policy="validation_only",
        result_scope="SCIENTIFIC",
    )


def case() -> EvaluationCase:
    return EvaluationCase(
        dataset="fixture",
        split="held_out",
        seed=1,
        expression=np.ones((3, 2), dtype=float),
        coordinates=np.ones((3, 2), dtype=float),
        labels=np.zeros(3, dtype=float),
    )


def evaluator(candidate: SpatialMethod, leakage=None) -> IntegrationEvaluator:
    project_config = project()
    return IntegrationEvaluator(
        project=project_config,
        design=design(),
        baselines={
            "method_a": ConstantMethod("method_a", 1.0),
            "method_b": ConstantMethod("method_b", 2.0),
        },
        candidate=candidate,
        ablations={
            "without_a": ConstantMethod("without_a", 1.2),
            "without_b": ConstantMethod("without_b", 1.3),
        },
        plan=plan(),
        leakage=leakage
        or audit_leakage([{"patient_id": "train"}], [{"patient_id": "test"}]),
        configuration_hash="fixture-configuration",
        implementation_kind="custom",
        code_commit="fixture-commit",
        claim_store=claims(project_config),
    )


def score(representation: FloatArray, labels: FloatArray) -> float:
    return float(representation.mean())


def test_underperforming_candidate_is_preserved_as_no_improvement() -> None:
    run = evaluator(ConstantMethod("candidate", 1.5)).evaluate((case(),), score)

    assert run.assessment.status == "NO_IMPROVEMENT"
    assert run.assessment.mean_margin_over_best_baseline == pytest.approx(-0.5)
    assert len(run.records) == 5
    assert {item.method for item in run.records} == {
        "method_a",
        "method_b",
        "candidate",
        "without_a",
        "without_b",
    }


def test_candidate_failure_is_preserved_and_assessment_is_inconclusive() -> None:
    run = evaluator(ConstantMethod("candidate", 0.0, fail=True)).evaluate((case(),), score)

    assert run.assessment.status == "INCONCLUSIVE"
    failed = next(item for item in run.records if item.method == "candidate")
    assert failed.status == "failed"
    assert "deliberate integration test failure" in failed.failure_reason


def test_leakage_blocks_before_any_adapter_executes() -> None:
    candidate = ConstantMethod("candidate", 3.0)
    blocked = audit_leakage([{"patient_id": "same"}], [{"patient_id": "same"}])
    integration = evaluator(candidate, blocked)

    with pytest.raises(ValueError, match="leakage audit is BLOCKED"):
        integration.evaluate((case(),), score)
    assert candidate.calls == 0


def test_design_must_use_every_algorithm_in_the_closed_set() -> None:
    incomplete = design().model_copy(update={"components": (design().components[0],)})
    with pytest.raises(ValueError, match="exactly the closed algorithm set"):
        incomplete.validate_for(project())


def test_scientific_evaluation_requires_audited_claim_store() -> None:
    project_config = project()
    with pytest.raises(ValueError, match="audited claim store"):
        IntegrationEvaluator(
            project=project_config,
            design=design(),
            baselines={
                "method_a": ConstantMethod("method_a", 1.0),
                "method_b": ConstantMethod("method_b", 2.0),
            },
            candidate=ConstantMethod("candidate", 3.0),
            ablations={
                "without_a": ConstantMethod("without_a", 1.2),
                "without_b": ConstantMethod("without_b", 1.3),
            },
            plan=plan(),
            leakage=audit_leakage(
                [{"patient_id": "train"}], [{"patient_id": "test"}]
            ),
            configuration_hash="fixture-configuration",
            implementation_kind="custom",
            code_commit="fixture-commit",
        )
