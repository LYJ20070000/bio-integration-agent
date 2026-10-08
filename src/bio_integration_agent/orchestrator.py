from __future__ import annotations

import json
import os
from pathlib import Path

from .contracts import EvidenceAuditReport, EvidenceBundle, ProjectConfig
from .evidence import EvidenceError, audit_evidence_bundle
from .evaluation import stable_hash
from .state import ProjectStage, StateMachine


class Orchestrator:
    def __init__(self, config: ProjectConfig, project_dir: Path) -> None:
        self.config = config
        self.project_dir = project_dir
        self.configuration_hash = stable_hash(config.model_dump(mode="json"))
        self.state = StateMachine(
            project_dir / "artifacts" / "project_state.json",
            context={
                "project_id": config.project_id,
                "configuration_hash": self.configuration_hash,
            },
        )

    @classmethod
    def from_json(cls, config_path: Path, project_dir: Path | None = None) -> "Orchestrator":
        config = ProjectConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
        return cls(config, project_dir or config_path.parent.parent)

    def validate(self) -> dict[str, object]:
        report: dict[str, object] = {
            "project_id": self.config.project_id,
            "direction": self.config.direction.value,
            "closed_algorithm_set": sorted(self.config.algorithm_ids),
            "configuration_hash": self.configuration_hash,
            "blocking_missing_fields": [],
            "non_blocking_missing_fields": self.config.missing_fields(),
        }
        path = self.project_dir / "artifacts" / "input_validation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if self.state.stage == ProjectStage.INITIALIZE:
            self.state.advance(ProjectStage.VALIDATE_INPUT, "Configuration validated")
        return report

    def status(self) -> dict[str, object]:
        return {
            "project_id": self.config.project_id,
            "configuration_hash": self.configuration_hash,
            "stage": self.state.stage.value,
            "history": self.state.history,
        }

    def write_evidence_template(self, path: Path) -> dict[str, object]:
        bundle = EvidenceBundle(
            project_id=self.config.project_id,
            configuration_hash=self.configuration_hash,
        )
        self._write_json(path, bundle.model_dump(mode="json"))
        return bundle.model_dump(mode="json")

    def import_evidence(self, path: Path) -> EvidenceAuditReport:
        bundle = EvidenceBundle.model_validate_json(path.read_text(encoding="utf-8"))
        ledger_path = self.project_dir / "artifacts" / "evidence_ledger.json"
        audit_path = self.project_dir / "artifacts" / "evidence_audit.json"

        if self.state.stage == ProjectStage.RETRIEVE_EVIDENCE:
            if ledger_path.exists() and audit_path.exists():
                existing_bundle = EvidenceBundle.model_validate_json(
                    ledger_path.read_text(encoding="utf-8")
                )
                existing_audit = EvidenceAuditReport.model_validate_json(
                    audit_path.read_text(encoding="utf-8")
                )
                if stable_hash(existing_bundle.model_dump(mode="json")) == stable_hash(
                    bundle.model_dump(mode="json")
                ):
                    return existing_audit
            raise EvidenceError("evidence is frozen after the workflow advances")
        if self.state.stage != ProjectStage.VALIDATE_INPUT:
            raise EvidenceError(
                "evidence import requires state VALIDATE_INPUT; run configuration validation first"
            )

        audit = audit_evidence_bundle(self.config, self.configuration_hash, bundle)
        self._write_json(audit_path, audit.model_dump(mode="json"))
        if audit.status != "PASS":
            raise EvidenceError(f"evidence audit is BLOCKED; inspect {audit_path}")

        self._write_json(ledger_path, bundle.model_dump(mode="json"))
        self.state.advance(
            ProjectStage.RETRIEVE_EVIDENCE,
            (
                f"Imported {audit.source_count} sources and {audit.claim_count} claims; "
                f"bundle_hash={audit.bundle_hash}"
            ),
        )
        return audit

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
