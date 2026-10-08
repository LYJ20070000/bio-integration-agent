from pathlib import Path

import pytest

from bio_integration_agent.state import ProjectStage, StateMachine


def test_state_persists_and_resumes(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = StateMachine(path)
    state.advance(ProjectStage.VALIDATE_INPUT, "valid")
    resumed = StateMachine(path)
    assert resumed.stage == ProjectStage.VALIDATE_INPUT
    assert resumed.history[-1]["note"] == "valid"


def test_state_rejects_skipping_steps(tmp_path: Path) -> None:
    state = StateMachine(tmp_path / "state.json")
    with pytest.raises(ValueError, match="invalid transition"):
        state.advance(ProjectStage.RUN_BENCHMARKS)


def test_state_rejects_different_project_context(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    StateMachine(path, context={"project_id": "one"})
    with pytest.raises(ValueError, match="context"):
        StateMachine(path, context={"project_id": "two"})


def test_terminal_state_cannot_resume_implicitly(tmp_path: Path) -> None:
    state = StateMachine(tmp_path / "state.json")
    state.advance(ProjectStage.BLOCKED, "blocked for test")
    with pytest.raises(ValueError, match="terminal"):
        state.advance(ProjectStage.VALIDATE_INPUT)
