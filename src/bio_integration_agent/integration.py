from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Literal, Mapping

import numpy as np

from .contracts import (
    EvaluationPlan,
    IntegrationAssessment,
    IntegrationDesign,
    LeakageReport,
    ProjectConfig,
    ResultRecord,
    ResultStatus,
    VerificationStatus,
)
from .evidence import ClaimStore
from .evaluation import stable_hash
from .methods import FloatArray, SpatialMethod


MetricFunction = Callable[[FloatArray, FloatArray], float]


@dataclass(frozen=True)
class EvaluationCase:
    dataset: str
    split: str
    seed: int
    expression: FloatArray
    coordinates: FloatArray
    labels: FloatArray
    image_features: FloatArray | None = None

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.dataset, self.split, self.seed)


@dataclass(frozen=True)
class IntegrationRun:
    records: tuple[ResultRecord, ...]
    assessment: IntegrationAssessment


class IntegrationEvaluator:
    """Run a declared integration fairly against its complete closed baseline set."""

    def __init__(
        self,
        *,
        project: ProjectConfig,
        design: IntegrationDesign,
        baselines: Mapping[str, SpatialMethod],
        candidate: SpatialMethod,
        ablations: Mapping[str, SpatialMethod],
        plan: EvaluationPlan,
        leakage: LeakageReport,
        configuration_hash: str,
        implementation_kind: Literal[
            "official", "mechanism_demonstration", "custom", "unverified"
        ],
        code_commit: str = "not_specified",
        claim_store: ClaimStore | None = None,
    ) -> None:
        self.project = project
        self.design = design
        self.baselines = dict(baselines)
        self.candidate = candidate
        self.ablations = dict(ablations)
        self.plan = plan
        self.leakage = leakage
        self.configuration_hash = configuration_hash
        self.implementation_kind = implementation_kind
        self.code_commit = code_commit
        self.claim_store = claim_store
        self._validate_setup()

    def _validate_setup(self) -> None:
        self.plan.validate_for(self.project)
        self.design.validate_for(self.project)
        if self.design.result_scope != self.plan.result_scope:
            raise ValueError("integration design and evaluation plan scopes do not match")
        if self.design.objective_metric not in self.plan.metrics:
            raise ValueError("integration objective metric is absent from the evaluation plan")
        if self.plan.result_scope == "SCIENTIFIC":
            if self.code_commit == "not_specified":
                raise ValueError("scientific integration evaluation requires a code_commit")
            if self.implementation_kind in {"unverified", "mechanism_demonstration"}:
                raise ValueError("scientific integration evaluation requires a verified implementation kind")
            if self.claim_store is None:
                raise ValueError("scientific integration evaluation requires an audited claim store")
            for component in self.design.components:
                for claim_id in component.evidence_claim_ids:
                    claim = self.claim_store.get_claim(claim_id)
                    if claim.algorithm_id != component.source_algorithm_id:
                        raise ValueError(
                            f"component '{component.component_id}' cites claim '{claim_id}' "
                            "from a different algorithm"
                        )
                    if claim.verification_status != VerificationStatus.VERIFIED:
                        raise ValueError(
                            f"component '{component.component_id}' cites an unverified claim"
                        )
        if set(self.baselines) != self.project.algorithm_ids:
            missing = sorted(self.project.algorithm_ids - set(self.baselines))
            extra = sorted(set(self.baselines) - self.project.algorithm_ids)
            raise ValueError(
                f"baseline adapters must match the closed algorithm set; missing={missing}, extra={extra}"
            )
        for method_id, method in (*self.baselines.items(), *self.ablations.items()):
            if method.method_id != method_id:
                raise ValueError(f"adapter key '{method_id}' does not match method_id '{method.method_id}'")
        if self.candidate.method_id != self.design.method_id:
            raise ValueError("candidate adapter method_id does not match the integration design")
        actual_integrated = {self.candidate.method_id, *self.ablations}
        if actual_integrated != set(self.plan.integrated_method_ids):
            raise ValueError("candidate and ablation adapters must exactly match integrated_method_ids")

    def evaluate(
        self,
        cases: tuple[EvaluationCase, ...],
        metric: MetricFunction,
    ) -> IntegrationRun:
        if self.leakage.status != "PASS":
            raise ValueError("integration evaluation blocked: leakage audit is BLOCKED")
        self._validate_cases(cases)
        plan_hash = stable_hash(self.plan.model_dump(mode="json"))
        leakage_hash = stable_hash(self.leakage.model_dump(mode="json"))
        methods = {
            **self.baselines,
            self.candidate.method_id: self.candidate,
            **self.ablations,
        }
        records: list[ResultRecord] = []
        for case in cases:
            for method_id in (*self.plan.baseline_method_ids, *self.plan.integrated_method_ids):
                method = methods[method_id]
                try:
                    representation = method.fit_transform(
                        case.expression, case.coordinates, case.image_features
                    )
                    value = float(metric(representation, case.labels))
                    record = ResultRecord(
                        method=method_id,
                        dataset=case.dataset,
                        split=case.split,
                        seed=case.seed,
                        configuration_hash=self.configuration_hash,
                        metric=self.design.objective_metric,
                        value=value,
                        code_commit=self.code_commit,
                        result_scope=self.plan.result_scope,
                        implementation_kind=self.implementation_kind,
                        evaluation_plan_hash=plan_hash,
                        leakage_report_hash=leakage_hash,
                    )
                except Exception as exc:  # A failed method is a result, not missing data.
                    record = ResultRecord(
                        method=method_id,
                        dataset=case.dataset,
                        split=case.split,
                        seed=case.seed,
                        configuration_hash=self.configuration_hash,
                        metric=self.design.objective_metric,
                        status=ResultStatus.FAILED,
                        failure_reason=f"{type(exc).__name__}: {exc}",
                        code_commit=self.code_commit,
                        result_scope=self.plan.result_scope,
                        implementation_kind=self.implementation_kind,
                        evaluation_plan_hash=plan_hash,
                        leakage_report_hash=leakage_hash,
                    )
                records.append(record)
        return IntegrationRun(tuple(records), self._assess(tuple(records), len(cases)))

    def _validate_cases(self, cases: tuple[EvaluationCase, ...]) -> None:
        keys = [case.key for case in cases]
        if len(keys) != len(set(keys)):
            raise ValueError("evaluation cases contain duplicate dataset/split/seed keys")
        expected = set(product(self.plan.datasets, self.plan.splits, self.plan.seeds))
        if set(keys) != expected:
            missing = sorted(expected - set(keys))
            extra = sorted(set(keys) - expected)
            raise ValueError(f"evaluation cases do not match the frozen plan; missing={missing}, extra={extra}")

    def _assess(
        self, records: tuple[ResultRecord, ...], planned: int
    ) -> IntegrationAssessment:
        by_key = {(item.dataset, item.split, item.seed, item.method): item for item in records}
        margins: list[float] = []
        for dataset, split, seed in product(
            self.plan.datasets, self.plan.splits, self.plan.seeds
        ):
            candidate = by_key[(dataset, split, seed, self.design.method_id)]
            baselines = [
                by_key[(dataset, split, seed, method_id)]
                for method_id in self.plan.baseline_method_ids
            ]
            if candidate.status != ResultStatus.SUCCESS or any(
                item.status != ResultStatus.SUCCESS for item in baselines
            ):
                continue
            baseline_values = [float(item.value) for item in baselines if item.value is not None]
            candidate_value = float(candidate.value) if candidate.value is not None else np.nan
            best = max(baseline_values) if self.design.higher_is_better else min(baseline_values)
            margin = candidate_value - best if self.design.higher_is_better else best - candidate_value
            margins.append(margin)

        mean_margin = float(np.mean(margins)) if margins else None
        worst_margin = float(np.min(margins)) if margins else None
        if self.plan.result_scope == "DEMO_ONLY":
            status = "DEMO_ONLY"
            reason = "Synthetic or mechanism-demonstration comparisons cannot support scientific superiority."
        elif len(margins) < planned or len(margins) < self.design.minimum_complete_comparisons:
            status = "INCONCLUSIVE"
            reason = "One or more planned candidate/baseline comparisons failed or are missing."
        elif mean_margin is not None and worst_margin is not None and (
            mean_margin >= self.design.minimum_improvement and worst_margin >= 0.0
        ):
            status = "IMPROVEMENT_SUPPORTED"
            reason = "The candidate met the predeclared improvement rule on complete comparisons."
        else:
            status = "NO_IMPROVEMENT"
            reason = "The candidate did not meet the predeclared improvement rule."
        return IntegrationAssessment(
            candidate_method_id=self.design.method_id,
            status=status,
            result_scope=self.plan.result_scope,
            metric=self.design.objective_metric,
            higher_is_better=self.design.higher_is_better,
            minimum_improvement=self.design.minimum_improvement,
            complete_comparisons=len(margins),
            planned_comparisons=planned,
            mean_margin_over_best_baseline=mean_margin,
            worst_margin_over_best_baseline=worst_margin,
            reason=reason,
        )
