# Quality Guidelines

> Code quality standards for backend development.

---

## Overview

This repository is maintained as a research codebase rather than a product service.

Quality checks are split into two layers:

- local authoring checks: structure, imports, unit tests, config sanity, lightweight verification,
- remote experiment checks: real training, large inference jobs, and metric-bearing runs on the remote server.

The local machine should not be treated as an experiment execution environment.
If a change needs GPU, large checkpoints, full datasets, or long-running evaluation, the
expected workflow is: commit, push, then run on the remote server.

---

## Forbidden Patterns

### Don't: Put heavy experiment logic in root launchers

Root launchers should orchestrate arguments and dispatch only.
Core logic belongs in `HYPIR/`.

Why it's bad:

- duplicates logic across scripts,
- makes experiment variants drift,
- makes remote reproduction harder.

### Don't: Hardcode machine-specific experiment paths in package modules

Dataset roots, checkpoint paths, and output directories must stay in configs or launch commands,
not inside reusable modules under `HYPIR/`.

Why it's bad:

- breaks remote reproducibility,
- leaks workstation-specific assumptions into shared code,
- makes commit-to-remote execution brittle.

### Don't: Claim experiment validation from a machine that cannot run the experiment

If local GPU/data/runtime are unavailable, do not present an unrun training or inference path
as validated.

Why it's bad:

- produces false confidence,
- hides the real boundary between code review and experiment verification.

### Don't: Mix baseline layers without naming the target path

This repo currently has multiple layers:

- original HYPIR-SD2 path,
- cascade deblur path,
- expert / MOE extensions.

Do not describe a result or code change as "HYPIR" without stating which layer it targets.

Why it's bad:

- makes ablations ambiguous,
- causes invalid comparisons across incompatible pipelines.

---

## Required Patterns

### Keep experiments config-driven

Training and inference variants should be selected through explicit config files or stage-specific
launchers.

Expected examples:

- `python train.py --config configs/sd2_train.yaml`
- `python train.py --config configs/sd2_cascade_blur.yaml`
- `python train_deblur.py --config configs/nafnet_finetune_blur.yaml`

### Keep local and remote responsibilities separate

Local work is for:

- code edits,
- spec updates,
- unit tests and smoke checks,
- command and config preparation.

Remote work is for:

- full training,
- dataset-scale inference,
- metric-bearing evaluation,
- long-running experiment monitoring.

### Record the exact remote launch surface

If a change affects experiment behavior, the final handoff should include:

- the config file,
- the launcher command,
- any required checkpoint or dataset assumptions,
- whether the run must happen remotely.

### Prefer lightweight local verification before handoff

Even without a local experiment environment, do the checks that are feasible:

- import / syntax safety,
- targeted unit tests under `tests/`,
- config-key consistency,
- launcher argument sanity.

---

## Scenario: Remote Experiment Execution Boundary

### 1. Scope / Trigger

- Trigger: the change touches training, inference, evaluation, checkpoints, dataset paths, or experiment configs.
- Trigger: the change requires GPU, large datasets, or remote-only runtime dependencies.

### 2. Signatures

Common launch surfaces in this repository:

- `python train.py --config <config>`
- `python train_deblur.py --config <config>`
- `python train_expert.py --config <config>`
- `python train_moe.py --config <config>`
- `python test.py ...`
- `python inference.py ...`
- `python inference_moe_hypir.py ...`

### 3. Contracts

- Local contract:
  the local machine is not assumed to have the full experiment environment.
- Remote contract:
  full experiments are expected to run after the relevant commits are pushed.
- Handoff contract:
  code and config must be committed and pushed before asking for a remote run against them.
- Reporting contract:
  if no remote run happened yet, state that clearly as `not remotely validated`.

### 4. Validation & Error Matrix

| Situation | Required Action | Do Not Do |
|-----------|-----------------|-----------|
| Code-only refactor | Run local checks if possible; note that no remote experiment was run | Claim training quality is unchanged without verification |
| Config update for training | Commit and push the config change; provide the exact remote command | Treat local file edits as the runnable source of truth |
| GPU-only feature | Verify imports/CLI locally if possible; defer performance claims to remote run | Block progress on unavailable local hardware |
| Missing local dataset/checkpoint | State limitation explicitly and hand off the remote launch surface | Fabricate a successful smoke run |

### 5. Good/Base/Bad Cases

#### Good

- Update `configs/sd2_cascade_blur.yaml`
- Commit and push
- Provide: `python train.py --config configs/sd2_cascade_blur.yaml`
- State that the real experiment must run on the remote server

#### Base

- Make trainer changes
- Run local unit tests or import checks only
- Report that remote validation is still pending

#### Bad

- Edit a trainer locally
- Skip commit/push
- Ask someone to run "the latest local version" on the server
- Report the path as verified without an actual remote run

### 6. Tests Required

Before remote handoff, do as many of these as the local environment allows:

- targeted unit tests in `tests/`
- import / module-load sanity
- `--help` or config parsing sanity for changed launchers
- consistency check between launcher arguments and config keys

Remote run should cover:

- actual training or inference execution,
- checkpoint writing and loading,
- result logging / metric logging if the change touches them.

### 7. Wrong vs Correct

#### Wrong

- "I changed `cascade_sd2.py`; please run it on the server."
- No commit hash
- No pushed branch
- No exact command
- No config reference

#### Correct

- "The change is on the pushed branch with the updated config."
- "Run `python train.py --config configs/sd2_cascade_blur.yaml` on the remote server."
- "Local machine has no experiment environment; remote validation is required."

---

## Testing Requirements

### Local expectations

- Run lightweight checks that match the actual local environment.
- Prefer narrow tests over pretending to do a full experiment locally.
- If no feasible local test exists, say so explicitly.

### Remote expectations

- Any result that depends on GPU throughput, full datasets, or checkpoint compatibility must be validated remotely.
- Training or large inference changes are not considered fully verified until they run on the remote server.

---

## Code Review Checklist

- Is the code placed in the correct layer: launcher vs `trainer/` vs `enhancer/` vs `model/` vs `dataset/`?
- Does the change clearly state which pipeline it targets: base SD2, cascade deblur, or expert/MOE?
- Are machine-specific paths kept in configs instead of reusable package code?
- Is the remote launch command explicit when experiment behavior changed?
- Does the summary clearly separate local verification from remote validation?
