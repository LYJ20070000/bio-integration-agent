"""Evidence-grounded algorithm integration agent."""

from .contracts import IntegrationDesign, ProjectConfig
from .integration import EvaluationCase, IntegrationEvaluator, IntegrationRun
from .orchestrator import Orchestrator

__all__ = [
    "EvaluationCase",
    "IntegrationDesign",
    "IntegrationEvaluator",
    "IntegrationRun",
    "Orchestrator",
    "ProjectConfig",
]
__version__ = "0.2.0"
