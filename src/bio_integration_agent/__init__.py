"""Evidence-grounded algorithm integration agent."""

from .contracts import ProjectConfig
from .orchestrator import Orchestrator

__all__ = ["Orchestrator", "ProjectConfig"]
__version__ = "0.1.0"
