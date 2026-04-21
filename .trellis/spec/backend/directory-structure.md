# Directory Structure

> How backend code is organized in this project.

---

## Overview

This repository is a research-first image restoration project.

The codebase follows a stable split:

- root-level scripts are thin launchers for training, inference, and demos,
- reusable logic lives under the `HYPIR/` package,
- experiment selection is config-driven through `configs/`,
- project process and AI-facing conventions live under `.trellis/`.

---

## Directory Layout

```text
.
├── HYPIR/
│   ├── dataset/      # Dataset loaders, labels, synthetic degradation pipeline
│   ├── enhancer/     # Inference-time restoration wrappers
│   ├── model/        # Core models, wrappers, discriminator, blur modules, MOE
│   ├── trainer/      # Training orchestration
│   └── utils/        # Shared helpers, EMA, captioning, tiled VAE, optimizers
├── configs/          # YAML experiment definitions
├── scripts/          # One-off dataset preparation helpers
├── tests/            # Lightweight regression/unit tests
├── examples/         # Sample inputs and prompts for inference
├── assets/           # Demo and README assets
├── .trellis/         # Workflow, tasks, and project specs
├── train.py          # Main SD2 / cascade training dispatcher
├── test.py           # Original SD2 batch inference entrypoint
├── app.py            # Gradio demo entrypoint
├── inference.py      # Simplified local batch inference wrapper
├── inference_moe_hypir.py
├── train_isblur.py
├── train_deblur.py
├── train_expert.py
└── train_moe.py
```

---

## Module Organization

### Root launchers stay thin

Root-level Python scripts should only:

- parse CLI arguments,
- load configs or checkpoint paths,
- select the correct trainer/enhancer,
- call into `HYPIR/...`.

They should not contain substantial model logic.

Examples:

- `train.py` dispatches by `base_model_type`
- `app.py` loads config and wraps `HYPIR.enhancer.sd2.SD2Enhancer`
- `inference.py` and `inference_moe_hypir.py` orchestrate I/O and call package modules

### `HYPIR/trainer/` owns training orchestration

Use `HYPIR/trainer/` for classes that define experiment lifecycle, optimizer setup,
validation, checkpointing, and cross-module coordination.

Examples:

- `HYPIR/trainer/base.py`: common SD2 training flow, VAE setup, discriminator, logging, validation
- `HYPIR/trainer/sd2.py`: SD2-specific scheduler, prompt encoding, LoRA UNet setup
- `HYPIR/trainer/cascade_sd2.py`: SD2 trainer with frozen deblur preprocessing
- `HYPIR/trainer/deblur_trainer.py`: standalone deblur stage
- `HYPIR/trainer/expert_trainer.py`, `HYPIR/trainer/moe_trainer.py`: specialist training stages

### `HYPIR/enhancer/` owns inference-time restoration

Use `HYPIR/enhancer/` for reusable inference wrappers that can be called by multiple
frontends.

Examples:

- `HYPIR/enhancer/base.py`: tiled VAE-aware enhancement flow
- `HYPIR/enhancer/sd2.py`: SD2 restoration wrapper used by demo and batch inference

### `HYPIR/model/` owns neural modules and wrappers

Use `HYPIR/model/` for pure model definitions or thin compatibility wrappers around them.

Examples:

- `HYPIR/model/D.py`: vision-aided discriminator
- `HYPIR/model/isblur.py`: blur classifier
- `HYPIR/model/nafnet.py`, `HYPIR/model/nafnet_wrapper.py`: deblur frontend
- `HYPIR/model/nafnet_moe.py`, `HYPIR/model/nafnet_moe_wrapper.py`: routing / expert composition
- `HYPIR/model/backbone.py`: OpenCLIP feature backbone used by discriminator

### `HYPIR/dataset/` owns data loading and degradation generation

Use `HYPIR/dataset/` for parquet-backed paired datasets, blur/expert filters, and
synthetic degradation logic.

Examples:

- `HYPIR/dataset/paired.py`: base paired parquet dataset
- `HYPIR/dataset/blur_labeled.py`: blur label extension
- `HYPIR/dataset/expert.py`: expert-type filtering
- `HYPIR/dataset/realesrgan.py`, `HYPIR/dataset/batch_transform.py`: degradation generation

### `configs/` is the experiment switchboard

Every experiment family should have an explicit YAML config.
Keep environment-specific paths and hyperparameters in config files, not in package code.

Examples:

- `configs/sd2_train.yaml`: base SD2 path
- `configs/sd2_cascade_blur.yaml`: cascade training path
- `configs/nafnet_finetune_blur.yaml`: deblur stage config
- `configs/nafnet_moe.yaml`: MOE routing config

### `scripts/` is for one-off data preparation, not core runtime

Only place helper utilities here when they are not imported as part of the main
training/inference path.

Examples:

- `scripts/add_degradation_label.py`
- `scripts/make_paired_meta.py`

---

## Naming Conventions

### File naming

- Use `snake_case.py` for package modules and standalone scripts.
- Name trainer variants by capability or training stage:
  `sd2.py`, `cascade_sd2.py`, `deblur_trainer.py`, `expert_trainer.py`, `moe_trainer.py`
- Name root launchers by user-facing task:
  `train_<stage>.py`, `inference_<variant>.py`, `app*.py`
- Name configs by model family + experiment intent:
  `sd2_train.yaml`, `sd2_cascade_blur.yaml`, `nafnet_finetune_blur.yaml`

### Placement rules

- If code is reusable across launchers, it belongs in `HYPIR/`, not at repo root.
- If logic is specific to experiment wiring, put it in `trainer/` or `enhancer/`.
- If logic defines a neural network block or wrapper, put it in `model/`.
- If logic is only for dataset generation or metadata conversion, put it in `scripts/`.

### Adding new work

When introducing a new experiment family:

1. Add the reusable module under `HYPIR/`.
2. Add or update a YAML config under `configs/`.
3. Add a root launcher only if the workflow is distinct enough to need a dedicated entrypoint.

---

## Examples

- `train.py` + `HYPIR/trainer/sd2.py`: thin launcher + trainer split
- `app.py` + `HYPIR/enhancer/sd2.py`: demo frontend delegating to reusable enhancer
- `HYPIR/trainer/cascade_sd2.py` + `configs/sd2_cascade_blur.yaml`: new experiment path expressed as trainer + config
- `train_deblur.py` + `HYPIR/trainer/deblur_trainer.py`: stage-specific launcher delegating to package code
