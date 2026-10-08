import pytest
from pydantic import ValidationError

from bio_integration_agent.contracts import (
    AlgorithmSpec,
    Direction,
    EvaluationPlan,
    ProjectConfig,
    ResultRecord,
)


def algorithms() -> tuple[AlgorithmSpec, ...]:
    return (
        AlgorithmSpec(algorithm_id="method_a", canonical_name="Method A"),
        AlgorithmSpec(algorithm_id="method_b", canonical_name="Method B"),
    )


def test_project_requires_two_unique_algorithms() -> None:
    with pytest.raises(ValidationError):
        ProjectConfig(project_id="demo", direction=Direction.SPATIAL_DOMAIN_IDENTIFICATION, algorithm_set=())


def test_algorithm_ids_are_frozen_as_a_set_view() -> None:
    project = ProjectConfig(
        project_id="demo",
        direction=Direction.SPATIAL_DOMAIN_IDENTIFICATION,
        algorithm_set=algorithms(),
    )
    assert project.algorithm_ids == frozenset({"method_a", "method_b"})
    with pytest.raises(ValidationError, match="frozen"):
        project.direction = Direction.SINGLE_CELL_INTEGRATION


def test_evaluation_plan_must_cover_exact_closed_set() -> None:
    project = ProjectConfig(
        project_id="demo",
        direction=Direction.SPATIAL_DOMAIN_IDENTIFICATION,
        algorithm_set=algorithms(),
    )
    plan = EvaluationPlan(
        baseline_method_ids=("method_a",),
        datasets=("dataset",),
        splits=("split",),
        metrics=("metric",),
        seeds=(1,),
        split_unit="patient",
        preprocessing_policy="train_only",
        hyperparameter_tuning_policy="validation_only",
        result_scope="SCIENTIFIC",
    )
    with pytest.raises(ValueError, match="exactly the closed-set baselines"):
        plan.validate_for(project)


def test_mechanism_demonstration_cannot_be_scientific() -> None:
    with pytest.raises(ValidationError, match="DEMO_ONLY"):
        ResultRecord(
            method="integrated_demo",
            dataset="synthetic",
            split="fixed",
            seed=1,
            configuration_hash="config-hash",
            metric="accuracy",
            value=1.0,
            code_commit="abc123",
            result_scope="SCIENTIFIC",
            implementation_kind="mechanism_demonstration",
        )
