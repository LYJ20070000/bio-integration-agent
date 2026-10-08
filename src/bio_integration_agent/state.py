from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any


class ProjectStage(StrEnum):
    INITIALIZE = "INITIALIZE"
    VALIDATE_INPUT = "VALIDATE_INPUT"
    RETRIEVE_EVIDENCE = "RETRIEVE_EVIDENCE"
    PROFILE_METHODS = "PROFILE_METHODS"
    ANALYZE_BIOLOGICAL_USAGE = "ANALYZE_BIOLOGICAL_USAGE"
    COMPARE_METHODS = "COMPARE_METHODS"
    GENERATE_INTEGRATION_HYPOTHESES = "GENERATE_INTEGRATION_HYPOTHESES"
    DESIGN_INTEGRATED_METHOD = "DESIGN_INTEGRATED_METHOD"
    DESIGN_EVALUATION = "DESIGN_EVALUATION"
    IMPLEMENT = "IMPLEMENT"
    UNIT_TEST = "UNIT_TEST"
    RUN_DEMO = "RUN_DEMO"
    RUN_BENCHMARKS = "RUN_BENCHMARKS"
    RUN_ABLATIONS = "RUN_ABLATIONS"
    ANALYZE_FAILURES = "ANALYZE_FAILURES"
    GENERATE_REPORT = "GENERATE_REPORT"
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


PIPELINE = (
    ProjectStage.INITIALIZE,
    ProjectStage.VALIDATE_INPUT,
    ProjectStage.RETRIEVE_EVIDENCE,
    ProjectStage.PROFILE_METHODS,
    ProjectStage.ANALYZE_BIOLOGICAL_USAGE,
    ProjectStage.COMPARE_METHODS,
    ProjectStage.GENERATE_INTEGRATION_HYPOTHESES,
    ProjectStage.DESIGN_INTEGRATED_METHOD,
    ProjectStage.DESIGN_EVALUATION,
    ProjectStage.IMPLEMENT,
    ProjectStage.UNIT_TEST,
    ProjectStage.RUN_DEMO,
    ProjectStage.RUN_BENCHMARKS,
    ProjectStage.RUN_ABLATIONS,
    ProjectStage.ANALYZE_FAILURES,
    ProjectStage.GENERATE_REPORT,
    ProjectStage.COMPLETE,
)


class StateMachine:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path, context: dict[str, Any] | None = None) -> None:
        self.path = path
        self.context = context or {}
        self.stage = ProjectStage.INITIALIZE
        self.history: list[dict[str, str]] = [
            {
                "stage": ProjectStage.INITIALIZE.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "note": "State machine initialized",
            }
        ]
        if path.exists():
            self._load()
        else:
            self.save()

    def advance(self, target: ProjectStage, note: str = "") -> None:
        terminal = {ProjectStage.BLOCKED, ProjectStage.PARTIAL, ProjectStage.FAILED}
        if self.stage in terminal or self.stage == ProjectStage.COMPLETE:
            raise ValueError(f"cannot advance terminal state {self.stage}")
        if target not in terminal:
            current_index = PIPELINE.index(self.stage)
            expected = PIPELINE[min(current_index + 1, len(PIPELINE) - 1)]
            if target != expected:
                raise ValueError(f"invalid transition {self.stage} -> {target}; expected {expected}")
        self.stage = target
        self.history.append(
            {
                "stage": target.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "note": note,
            }
        )
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "context": self.context,
                    "stage": self.stage.value,
                    "history": self.history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported or missing state schema_version")
        if data.get("context", {}) != self.context:
            raise ValueError("persisted state context does not match the current project")
        self.stage = ProjectStage(data["stage"])
        self.history = data["history"]
        if not self.history or self.history[-1].get("stage") != self.stage.value:
            raise ValueError("persisted state history does not match current stage")
