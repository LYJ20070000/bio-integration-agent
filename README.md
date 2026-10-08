# Bio Integration Agent

An evidence-grounded framework for designing and evaluating integrations among a user-supplied, closed set of biological algorithms.

This first checkpoint implements the safety-critical core: immutable typed project contracts, closed-set enforcement, claim provenance, evidence-linked integration components, a persisted state machine bound to the configuration hash, frozen evaluation plans, leakage-gated result storage, failure-preserving result records, method interfaces, a CLI, and tests.

## Example scope

The included example uses one direction, `spatial_domain_identification`, and exactly two candidate methods: SpaGCN and STAGATE. The example was selected because their published mechanisms suggest a testable combination of histology-aware spatial graph construction and adaptive graph attention.

The executable demo is intentionally a **mechanism demonstration**, not a reimplementation or scientific benchmark of either package. Its synthetic scores are labeled `DEMO_ONLY` and cannot support performance claims.

Primary references:

- SpaGCN paper: https://www.nature.com/articles/s41592-021-01255-8
- SpaGCN repository: https://github.com/jianhuupenn/SpaGCN
- STAGATE paper: https://www.nature.com/articles/s41467-022-29439-6
- STAGATE repository: https://github.com/zhanglabtools/STAGATE

## Install and run

```powershell
cd D:\Aleph\bio
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\bio-integration-agent validate configs\spatial_demo.json
.venv\Scripts\bio-integration-agent status configs\spatial_demo.json
.venv\Scripts\bio-integration-agent evidence-template configs\spatial_demo.json --output artifacts\evidence_template.json
.venv\Scripts\bio-integration-agent demo --config configs\spatial_demo.json --output artifacts\demo
.venv\Scripts\python -m pytest --basetemp=.pytest_tmp
```

Use a standard CPython 3.12 or 3.13 installation when available. The project is also verified with standard Windows CPython 3.14.7; an MSYS Python installation is not a substitute because native PyPI wheels generally do not target that platform.

The demo writes the following under `artifacts/demo`:

- `evaluation_plan.json`: the policy frozen before scoring, including the exact baseline set, data, split, metric, seed, preprocessing, tuning, and scope.
- `leakage_report.json`: the blocking leakage decision and findings.
- `results.json`: baseline, integrated-demo, and ablation outcomes. Every record carries configuration, evaluation-plan, and leakage-report provenance and remains `DEMO_ONLY`.
- `demo_summary.json`: a concise non-scientific summary with baseline and ablation coverage.
- `report.md`: a human-readable traceability report with the critical provenance hashes and scope warning.
- `manifest.json`: SHA-256 checksums for the generated artifacts.

The demo replaces its result artifact on each invocation, so rerunning the same command does not silently append duplicate observations.

## Evidence handoff

`evidence-template` writes an empty bundle containing the exact project identifier and configuration hash. It intentionally contains no sources or claims. Populate it only from material you actually inspected, then run:

```powershell
.venv\Scripts\bio-integration-agent import-evidence configs\spatial_demo.json artifacts\evidence_bundle.json
```

Import applies the following gates before the persisted workflow can advance from `VALIDATE_INPUT` to `RETRIEVE_EVIDENCE`:

- The bundle must match the exact immutable project configuration.
- Source and claim identifiers must be unique. Every claim must reference a registered source and an in-set algorithm.
- Claim source classifications must agree with their registered sources. A verified claim must use a verified source.
- Supporting claims must appear earlier in the bundle. Agent inferences require supporting claims and cannot be labeled as verified source facts.
- Every supplied algorithm needs at least one verified, non-inference claim from an original paper, official repository, or official documentation.
- Missing requested independent-benchmark or biological-application claims produce explicit warnings, not invented evidence and not assertions that such evidence does not exist.

A rejected import writes `artifacts/evidence_audit.json`, leaves the state unchanged, and does not create an accepted ledger. A passing import also writes `artifacts/evidence_ledger.json`, records the bundle hash in state history, and freezes the evidence stage against silent replacement. Reimporting the identical bundle is idempotent.

## Scientific boundaries

- `project.algorithm_set` is validated once and treated as the only candidate set.
- Claims for algorithms outside that set are rejected by `ClaimStore`.
- Agent inferences require supporting claim identifiers.
- Claim source classifications must match their registered source, and verified claims require verified sources.
- Integration components require an in-set source algorithm and evidence claims.
- Evaluation plans must include exactly every supplied baseline before execution.
- Empty partitions, missing split identifiers, identifier overlap, non-training-only preprocessing, test-label feature selection, and test-set tuning block results.
- Results are rejected when the leakage report is blocked. Failed and pending outcomes have explicit schemas; no code assumes the integrated method must win.
- `SCIENTIFIC` result records require a code commit and a declared implementation kind. The included implementations always emit `DEMO_ONLY`.

## State and recovery

`artifacts/project_state.json` is created when an orchestrator command starts. It contains a schema version, ordered transition history, project identifier, and configuration hash. Reusing that state file with a changed configuration is rejected instead of silently resuming the wrong project. Transitions must follow the declared pipeline; blocked, partial, failed, and complete states cannot be resumed implicitly. Configuration validation and evidence import are currently wired to their respective state transitions.

## Current limitations

- SpaGCN and STAGATE adapters are mechanism demonstrations only; official packages, pinned versions, and official preprocessing are not integrated.
- The research state machine defines the full ordered workflow, but method profiling and later scientific stages are not yet wired. The standalone synthetic demo does not claim to complete those stages.
- No literature retrieval has been performed and no accepted scientific evidence ledger is bundled. URLs in the example configuration are user-supplied identifiers, not claims that this agent independently verified them.
- The leakage audit checks declared identifiers and policies; it cannot discover undeclared biological relatedness, duplicated raw material, or preprocessing leakage without direction-specific adapters and metadata.
- The synthetic generator, label-based metric, and ablation outputs are software checks only. They are not biological datasets, benchmarks, or evidence of superiority.
- Resource estimation, real dataset acquisition, hyperparameter search, official baseline execution, and direction-specific scientific report templates remain future adapters.

## Next implementation layer

Real projects need direction-specific plugins for evidence retrieval, official baseline execution, datasets, metrics, resource estimates, and report templates. Those adapters should implement the shared contracts without weakening the invariants. Official implementations must replace the mechanism demo before scientific benchmarking.
