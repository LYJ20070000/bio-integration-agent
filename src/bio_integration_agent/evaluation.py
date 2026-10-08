from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import LeakageFinding, LeakageReport, ResultRecord


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_leakage(
    train: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    *,
    split_keys: tuple[str, ...] = ("patient_id", "sample_id", "section_id"),
    preprocessing_fit_scope: str = "train_only",
    feature_selection_used_test_labels: bool = False,
    hyperparameters_tuned_on_test: bool = False,
) -> LeakageReport:
    findings: list[LeakageFinding] = []
    if not train:
        findings.append(
            LeakageFinding(
                check="train_partition",
                severity="blocking",
                detail="training partition is empty",
            )
        )
    if not test:
        findings.append(
            LeakageFinding(
                check="test_partition",
                severity="blocking",
                detail="test partition is empty",
            )
        )
    usable_split_keys = tuple(
        key
        for key in split_keys
        if any(row.get(key) is not None for row in train)
        and any(row.get(key) is not None for row in test)
    )
    if not usable_split_keys:
        findings.append(
            LeakageFinding(
                check="split_identifiers",
                severity="blocking",
                detail="no configured split identifier is present in both partitions",
            )
        )
    for key in split_keys:
        train_ids = {row.get(key) for row in train if row.get(key) is not None}
        test_ids = {row.get(key) for row in test if row.get(key) is not None}
        overlap = train_ids & test_ids
        if overlap:
            findings.append(
                LeakageFinding(
                    check=f"{key}_overlap",
                    severity="blocking",
                    detail=f"overlapping identifiers: {sorted(map(str, overlap))}",
                )
            )
    if preprocessing_fit_scope != "train_only":
        findings.append(
            LeakageFinding(
                check="preprocessing_fit_scope",
                severity="blocking",
                detail=f"expected train_only, got {preprocessing_fit_scope}",
            )
        )
    if feature_selection_used_test_labels:
        findings.append(
            LeakageFinding(
                check="feature_selection",
                severity="blocking",
                detail="test labels were used for feature selection",
            )
        )
    if hyperparameters_tuned_on_test:
        findings.append(
            LeakageFinding(
                check="hyperparameter_tuning",
                severity="blocking",
                detail="test data were used for hyperparameter tuning",
            )
        )
    blocked = any(item.severity == "blocking" for item in findings)
    return LeakageReport(status="BLOCKED" if blocked else "PASS", findings=tuple(findings))


class ResultStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, result: ResultRecord, leakage: LeakageReport) -> None:
        self._validate(result, leakage)
        records = self.read()
        records.append(result)
        self._write(records)

    def replace(self, results: Sequence[ResultRecord], leakage: LeakageReport) -> None:
        for result in results:
            self._validate(result, leakage)
        self._write(list(results))

    def read(self) -> list[ResultRecord]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("result artifact must contain a JSON list")
        return [ResultRecord.model_validate(item) for item in data]

    def _validate(self, result: ResultRecord, leakage: LeakageReport) -> None:
        if leakage.status != "PASS":
            raise ValueError("benchmark result rejected: leakage audit is BLOCKED")
        expected_hash = stable_hash(leakage.model_dump(mode="json"))
        if result.leakage_report_hash != expected_hash:
            raise ValueError("result leakage_report_hash does not match the supplied audit")
        if result.evaluation_plan_hash == "not_specified":
            raise ValueError("accepted results require evaluation_plan_hash provenance")

    def _write(self, records: Sequence[ResultRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps([item.model_dump(mode="json") for item in records], indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
