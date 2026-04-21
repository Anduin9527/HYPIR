# HYPIR Paper-Aligned Structure Analysis

## Goal

Initialize a Trellis task that maps this repository against the HYPIR paper
([arXiv:2507.20590](https://arxiv.org/abs/2507.20590)) and clarifies the actual
project structure after local deblurring-oriented modifications.

## Requirements

- Produce a task-local analysis document that explains the repository layout.
- Separate the paper-native restoration path from later local extensions.
- Identify the main training, inference, model, dataset, and config entry points.
- Highlight the parts that matter for subsequent diffusion-based deblurring work.
- Record the likely files to revisit in future experiment tasks.

## Acceptance Criteria

- [x] `research.md` exists under this task directory.
- [x] The analysis explains the top-level execution flow from CLI/config to core modules.
- [x] The analysis distinguishes paper method assumptions from the current code reality.
- [x] The analysis explicitly identifies the deblurring-related extensions added on top of the original repo.
- [x] The task is initialized with Trellis context files and can be resumed as the current task.

## Technical Notes

- This is an analysis-first task; no production code changes are required.
- The repository behaves as a modified research sandbox rather than a paper-faithful reproduction.
- Future implementation tasks should branch from this document instead of re-discovering structure from scratch.
