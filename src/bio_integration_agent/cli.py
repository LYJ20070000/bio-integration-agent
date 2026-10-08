from __future__ import annotations

import argparse
import json
from pathlib import Path

from .demo import run_demo
from .orchestrator import Orchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bio-integration-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate project configuration")
    validate.add_argument("config", type=Path)

    status = subparsers.add_parser("status", help="show persisted workflow state")
    status.add_argument("config", type=Path)

    evidence_template = subparsers.add_parser(
        "evidence-template", help="write an empty project-bound evidence bundle"
    )
    evidence_template.add_argument("config", type=Path)
    evidence_template.add_argument(
        "--output", type=Path, default=Path("artifacts/evidence_template.json")
    )

    evidence_import = subparsers.add_parser(
        "import-evidence", help="validate and freeze a user-supplied evidence bundle"
    )
    evidence_import.add_argument("config", type=Path)
    evidence_import.add_argument("bundle", type=Path)

    demo = subparsers.add_parser("demo", help="run the non-scientific synthetic demonstration")
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--config", type=Path, default=Path("configs/spatial_demo.json"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "validate":
            agent = Orchestrator.from_json(args.config)
            print(json.dumps(agent.validate(), indent=2))
        elif args.command == "status":
            agent = Orchestrator.from_json(args.config)
            print(json.dumps(agent.status(), indent=2))
        elif args.command == "evidence-template":
            agent = Orchestrator.from_json(args.config)
            print(json.dumps(agent.write_evidence_template(args.output), indent=2))
        elif args.command == "import-evidence":
            agent = Orchestrator.from_json(args.config)
            report = agent.import_evidence(args.bundle)
            print(report.model_dump_json(indent=2))
        elif args.command == "demo":
            print(json.dumps(run_demo(args.output, args.seed, args.config), indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
