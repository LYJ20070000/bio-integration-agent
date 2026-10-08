from __future__ import annotations

from .contracts import IntegrationComponent, ProjectConfig
from .evidence import ClaimStore


class ComponentRegistry:
    def __init__(self, project: ProjectConfig, claims: ClaimStore) -> None:
        self._allowed_algorithms = project.algorithm_ids
        self._claims = claims
        self._components: dict[str, IntegrationComponent] = {}

    def register(self, component: IntegrationComponent) -> None:
        if component.source_algorithm_id not in self._allowed_algorithms:
            raise ValueError(
                f"component source '{component.source_algorithm_id}' is outside the closed set"
            )
        if not component.evidence_claim_ids:
            raise ValueError("integration components require evidence_claim_ids")
        for claim_id in component.evidence_claim_ids:
            self._claims.get_claim(claim_id)
        if component.component_id in self._components:
            raise ValueError(f"duplicate component_id: {component.component_id}")
        self._components[component.component_id] = component

    def all(self) -> tuple[IntegrationComponent, ...]:
        return tuple(self._components.values())

    def required_ablations(self) -> tuple[str, ...]:
        return tuple(component.ablation_flag for component in self._components.values())
