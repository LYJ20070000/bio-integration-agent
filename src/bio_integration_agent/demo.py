from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import EvaluationPlan, IntegrationComponent, IntegrationDesign, ProjectConfig
from .evaluation import ResultStore, audit_leakage, stable_hash
from .integration import EvaluationCase, IntegrationEvaluator
from .methods import IntegratedMechanismDemo, STAGATEMechanismDemo, SpaGCNMechanismDemo


def _synthetic_tissue(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    side = 10
    coordinates = np.array([(x, y) for x in range(side) for y in range(side)], dtype=float)
    labels = (coordinates[:, 0] >= side / 2).astype(int)
    expression = rng.normal(0, 0.45, size=(side * side, 8))
    expression[:, :4] += labels[:, None] * 1.8
    expression[:, 4:] += (1 - labels[:, None]) * 1.8
    image_features = labels[:, None] + rng.normal(0, 0.25, size=(side * side, 1))
    return expression, coordinates, image_features, labels


def _two_cluster_accuracy(representation: np.ndarray, labels: np.ndarray) -> float:
    # Deterministic two-means; reference labels are used only after clustering.
    centered = representation - representation.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ axes[0]
    centroids = np.array([projection.min(), projection.max()], dtype=float)
    assignments = np.zeros(projection.shape[0], dtype=int)
    for _ in range(50):
        updated = np.abs(projection[:, None] - centroids[None, :]).argmin(axis=1)
        if np.array_equal(updated, assignments):
            break
        assignments = updated
        centroids = np.array(
            [
                projection[assignments == index].mean()
                if np.any(assignments == index)
                else centroids[index]
                for index in (0, 1)
            ]
        )
    direct = float((assignments == labels).mean())
    return max(direct, 1.0 - direct)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_demo(
    output_dir: Path,
    seed: int = 7,
    config_path: Path = Path("configs/spatial_demo.json"),
) -> dict[str, object]:
    project = ProjectConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    expression, coordinates, image_features, labels = _synthetic_tissue(seed)
    train = [{"patient_id": "demo_train", "section_id": "section_a"}]
    test = [{"patient_id": "demo_test", "section_id": "section_b"}]
    leakage = audit_leakage(train, test)
    output_dir.mkdir(parents=True, exist_ok=True)
    leakage_path = output_dir / "leakage_report.json"
    leakage_path.write_text(
        leakage.model_dump_json(indent=2), encoding="utf-8"
    )

    baselines = (SpaGCNMechanismDemo(), STAGATEMechanismDemo())
    integrated = IntegratedMechanismDemo()
    ablations = (
        IntegratedMechanismDemo(use_histology_graph=False),
        IntegratedMechanismDemo(use_attention=False),
        IntegratedMechanismDemo(use_histology_graph=False, use_attention=False),
    )
    methods = (*baselines, integrated, *ablations)
    plan = EvaluationPlan(
        baseline_method_ids=tuple(method.method_id for method in baselines),
        integrated_method_ids=tuple(method.method_id for method in methods[len(baselines) :]),
        datasets=("synthetic_two_domain_tissue",),
        splits=("fixed_demo_split",),
        metrics=("two_cluster_label_accuracy",),
        seeds=(seed,),
        split_unit="synthetic_section",
        preprocessing_policy="identical generated inputs; no fitted preprocessing",
        hyperparameter_tuning_policy="none",
        result_scope="DEMO_ONLY",
    )
    plan.validate_for(project)
    plan_path = output_dir / "evaluation_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    plan_hash = stable_hash(plan.model_dump(mode="json"))
    leakage_hash = stable_hash(leakage.model_dump(mode="json"))
    config = {
        "seed": seed,
        "dataset": "synthetic_two_domain_tissue",
        "generator": {"side": 10, "expression_features": 8},
        "scope": "DEMO_ONLY",
    }
    config_hash = stable_hash(config)
    design = IntegrationDesign(
        method_id=integrated.method_id,
        components=(
            IntegrationComponent(
                component_id="histology_graph",
                source_algorithm_id="spagcn",
                source_component="histology-aware spatial graph mechanism",
                adapted_component="demonstration graph smoothing stage",
                target_weakness="spatial representation without demonstrated image context",
                evidence_claim_ids=(),
                new_assumptions=("Unverified mechanism mapping; DEMO_ONLY.",),
                ablation_flag="ablation_without_histology_graph",
            ),
            IntegrationComponent(
                component_id="graph_attention",
                source_algorithm_id="stagate",
                source_component="adaptive graph attention mechanism",
                adapted_component="demonstration attention smoothing stage",
                target_weakness="fixed graph weights in the preceding stage",
                evidence_claim_ids=(),
                new_assumptions=("Sequential composition is unverified; DEMO_ONLY.",),
                ablation_flag="ablation_without_attention",
            ),
        ),
        composition_description=(
            "Apply the SpaGCN-inspired demonstration graph transform followed by the "
            "STAGATE-inspired demonstration attention transform."
        ),
        objective_metric="two_cluster_label_accuracy",
        minimum_improvement=0.0,
        minimum_complete_comparisons=1,
        result_scope="DEMO_ONLY",
    )
    evaluator = IntegrationEvaluator(
        project=project,
        design=design,
        baselines={method.method_id: method for method in baselines},
        candidate=integrated,
        ablations={method.method_id: method for method in ablations},
        plan=plan,
        leakage=leakage,
        configuration_hash=config_hash,
        implementation_kind="mechanism_demonstration",
    )
    run = evaluator.evaluate(
        (
            EvaluationCase(
                dataset="synthetic_two_domain_tissue",
                split="fixed_demo_split",
                seed=seed,
                expression=expression,
                coordinates=coordinates,
                image_features=image_features,
                labels=labels,
            ),
        ),
        _two_cluster_accuracy,
    )
    store = ResultStore(output_dir / "results.json")
    store.replace(run.records, leakage)
    scores = {item.method: item.value for item in run.records}
    assessment_path = output_dir / "integration_assessment.json"
    assessment_path.write_text(run.assessment.model_dump_json(indent=2), encoding="utf-8")

    summary: dict[str, object] = {
        "status": "DEMO_ONLY",
        "warning": "Mechanism demonstrations are not official algorithm implementations or scientific benchmarks.",
        "leakage_status": leakage.status,
        "baseline_coverage": [method.method_id for method in baselines],
        "required_baselines": sorted(project.algorithm_ids),
        "ablation_methods": [method.method_id for method in ablations],
        "scores": scores,
        "integration_assessment": run.assessment.model_dump(mode="json"),
        "conclusion": "No scientific performance conclusion may be drawn from synthetic demo results.",
    }
    summary_path = output_dir / "demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = output_dir / "report.md"
    report_path.write_text(
        "# Synthetic mechanism demonstration\n\n"
        "**Scope: DEMO_ONLY.** These are mechanism demonstrations, not official SpaGCN "
        "or STAGATE reproductions, and the synthetic scores support no scientific "
        "performance claim.\n\n"
        f"- Leakage audit: `{leakage.status}`\n"
        f"- Closed-set baselines evaluated: `{', '.join(plan.baseline_method_ids)}`\n"
        f"- Ablations evaluated: `{', '.join(method.method_id for method in ablations)}`\n"
        f"- Evaluation-plan hash: `{plan_hash}`\n"
        f"- Leakage-report hash: `{leakage_hash}`\n\n"
        f"- Candidate assessment: `{run.assessment.status}`\n"
        f"- Observed mean margin over the best baseline: "
        f"`{run.assessment.mean_margin_over_best_baseline}`\n\n"
        "Machine-readable records are in `results.json`; failed outcomes, if any, are "
        "retained there with their error reason.\n",
        encoding="utf-8",
    )
    artifact_paths = (
        leakage_path,
        plan_path,
        store.path,
        assessment_path,
        summary_path,
        report_path,
    )
    manifest = {
        "scope": "DEMO_ONLY",
        "project_id": project.project_id,
        "configuration_hash": stable_hash(project.model_dump(mode="json")),
        "artifacts": {path.name: _file_sha256(path) for path in artifact_paths},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary
