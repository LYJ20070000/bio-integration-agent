from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class SpatialMethod(ABC):
    method_id: str

    @abstractmethod
    def fit_transform(
        self,
        expression: FloatArray,
        coordinates: FloatArray,
        image_features: FloatArray | None = None,
    ) -> FloatArray:
        """Return a representation suitable for downstream clustering."""

    def metadata(self) -> dict[str, Any]:
        return {"method_id": self.method_id, "implementation": type(self).__name__}


def pairwise_distances(values: FloatArray) -> FloatArray:
    delta = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def normalized_adjacency(weights: FloatArray) -> FloatArray:
    weights = weights.copy()
    np.fill_diagonal(weights, 1.0)
    degree = np.maximum(weights.sum(axis=1), 1e-12)
    scale = 1.0 / np.sqrt(degree)
    return scale[:, None] * weights * scale[None, :]


class SpaGCNMechanismDemo(SpatialMethod):
    """Mechanism demo only; this is not the official SpaGCN implementation."""

    method_id = "spagcn"

    def fit_transform(
        self,
        expression: FloatArray,
        coordinates: FloatArray,
        image_features: FloatArray | None = None,
    ) -> FloatArray:
        spatial = pairwise_distances(coordinates)
        scale = max(float(np.median(spatial[spatial > 0])), 1e-12)
        weights = np.exp(-(spatial / scale) ** 2)
        if image_features is not None:
            image = pairwise_distances(image_features)
            image_scale = max(float(np.median(image[image > 0])), 1e-12)
            weights *= np.exp(-0.5 * (image / image_scale) ** 2)
        return normalized_adjacency(weights) @ expression


class STAGATEMechanismDemo(SpatialMethod):
    """Mechanism demo only; this is not the official STAGATE implementation."""

    method_id = "stagate"

    def fit_transform(
        self,
        expression: FloatArray,
        coordinates: FloatArray,
        image_features: FloatArray | None = None,
    ) -> FloatArray:
        spatial = pairwise_distances(coordinates)
        radius = max(float(np.quantile(spatial[spatial > 0], 0.2)), 1e-12)
        neighborhood = (spatial <= radius).astype(float)
        normalized = expression / np.maximum(np.linalg.norm(expression, axis=1, keepdims=True), 1e-12)
        similarity = np.maximum(normalized @ normalized.T, 0.0)
        attention = normalized_adjacency(neighborhood * similarity)
        return 0.5 * expression + 0.5 * (attention @ expression)


class IntegratedMechanismDemo(SpatialMethod):
    """Synthetic demonstrator combining only mechanisms traced to the closed set."""

    def __init__(self, *, use_histology_graph: bool = True, use_attention: bool = True) -> None:
        self.use_histology_graph = use_histology_graph
        self.use_attention = use_attention
        if use_histology_graph and use_attention:
            self.method_id = "spagcn_stagate_integrated_demo"
        elif use_attention:
            self.method_id = "ablation_without_histology_graph"
        elif use_histology_graph:
            self.method_id = "ablation_without_attention"
        else:
            self.method_id = "ablation_without_both_components"

    def fit_transform(
        self,
        expression: FloatArray,
        coordinates: FloatArray,
        image_features: FloatArray | None = None,
    ) -> FloatArray:
        representation = expression
        if self.use_histology_graph:
            representation = SpaGCNMechanismDemo().fit_transform(
                representation, coordinates, image_features
            )
        if self.use_attention:
            representation = STAGATEMechanismDemo().fit_transform(
                representation, coordinates, image_features
            )
        return representation

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "use_histology_graph": self.use_histology_graph,
                "use_attention": self.use_attention,
                "scope": "DEMO_ONLY",
            }
        )
        return metadata
