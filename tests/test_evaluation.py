from pathlib import Path

import pytest

from bio_integration_agent.contracts import ResultRecord, ResultStatus
from bio_integration_agent.evaluation import ResultStore, audit_leakage, stable_hash


def test_patient_overlap_blocks_benchmark() -> None:
    report = audit_leakage(
        [{"patient_id": "p1", "section_id": "a"}],
        [{"patient_id": "p1", "section_id": "b"}],
    )
    assert report.status == "BLOCKED"
    assert report.findings[0].check == "patient_id_overlap"


def test_result_store_rejects_blocked_result(tmp_path: Path) -> None:
    report = audit_leakage([{"patient_id": "p1"}], [{"patient_id": "p1"}])
    store = ResultStore(tmp_path / "results.json")
    result = ResultRecord(
        method="method_a",
        dataset="demo",
        split="fixed",
        seed=1,
        configuration_hash="abc",
        metric="accuracy",
        value=0.5,
    )
    with pytest.raises(ValueError, match="leakage audit"):
        store.append(result, report)


def test_missing_split_identifiers_block_audit() -> None:
    report = audit_leakage([{"batch": "a"}], [{"batch": "b"}])
    assert report.status == "BLOCKED"
    assert any(item.check == "split_identifiers" for item in report.findings)


def test_failed_result_is_preserved_with_provenance(tmp_path: Path) -> None:
    report = audit_leakage([{"patient_id": "p1"}], [{"patient_id": "p2"}])
    report_hash = stable_hash(report.model_dump(mode="json"))
    result = ResultRecord(
        method="method_a",
        dataset="demo",
        split="fixed",
        seed=1,
        configuration_hash="config-hash",
        metric="accuracy",
        status=ResultStatus.FAILED,
        failure_reason="RuntimeError: deliberate test failure",
        evaluation_plan_hash="plan-hash",
        leakage_report_hash=report_hash,
    )
    store = ResultStore(tmp_path / "results.json")
    store.append(result, report)
    assert store.read() == [result]
