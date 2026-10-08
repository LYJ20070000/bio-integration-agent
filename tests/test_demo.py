import json
from pathlib import Path

from bio_integration_agent.demo import run_demo


def test_demo_evaluates_all_baselines_and_writes_traceable_artifacts(tmp_path: Path) -> None:
    config = Path(__file__).parents[1] / "configs" / "spatial_demo.json"
    output = tmp_path / "demo"
    summary = run_demo(output, seed=7, config_path=config)

    assert summary["status"] == "DEMO_ONLY"
    assert set(summary["baseline_coverage"]) == {"spagcn", "stagate"}
    assert len(summary["ablation_methods"]) == 3
    for name in (
        "leakage_report.json",
        "evaluation_plan.json",
        "results.json",
        "integration_assessment.json",
        "demo_summary.json",
        "report.md",
        "manifest.json",
    ):
        assert (output / name).is_file()

    records = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert len(records) == 6
    assert all(item["result_scope"] == "DEMO_ONLY" for item in records)
    assert all(item["implementation_kind"] == "mechanism_demonstration" for item in records)
    assert all(item["leakage_report_hash"] != "not_specified" for item in records)
    assert summary["integration_assessment"]["status"] == "DEMO_ONLY"
