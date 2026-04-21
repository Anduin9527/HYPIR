# HYPIR Structure Research

## Relevant Specs

- `.trellis/spec/backend/index.md`: repository-level backend guidance index; currently bootstrap-level only.
- `.trellis/spec/guides/index.md`: shared thinking guidance for cross-layer changes and reuse checks.

## Code Patterns Found

- Thin root CLI entrypoints dispatch into package modules.
  Example files: `train.py`, `test.py`, `app.py`, `inference.py`
- Training variants live under `HYPIR/trainer/` and either extend `BaseTrainer` or provide standalone stage trainers.
  Example files: `HYPIR/trainer/sd2.py`, `HYPIR/trainer/cascade_sd2.py`, `HYPIR/trainer/deblur_trainer.py`
- Inference wrappers live under `HYPIR/enhancer/` and expose tiled, prompt-conditioned restoration over SD2.
  Example files: `HYPIR/enhancer/base.py`, `HYPIR/enhancer/sd2.py`
- Configuration is YAML-driven and selects trainer/data behavior via `base_model_type` and `target` fields.
  Example files: `configs/sd2_train.yaml`, `configs/sd2_cascade_blur.yaml`, `configs/nafnet_finetune_blur.yaml`

## Files To Modify

- `.trellis/tasks/04-21-hypir-paper-structure-analysis/prd.md`: task scope and acceptance criteria.
- `.trellis/tasks/04-21-hypir-paper-structure-analysis/research.md`: project structure analysis and paper alignment.
- Likely future experiment touchpoints:
  `HYPIR/trainer/cascade_sd2.py`, `HYPIR/trainer/deblur_trainer.py`,
  `HYPIR/model/nafnet_wrapper.py`, `HYPIR/model/nafnet_moe.py`,
  `configs/sd2_cascade_blur.yaml`, `configs/nafnet_finetune_blur.yaml`,
  `inference.py`, `inference_moe_hypir.py`

## Paper Summary

The paper presents HYPIR as a diffusion-based image restoration framework that:

- leverages diffusion-yielded score priors for restoration,
- performs restoration in a single-step latent-space forward pass,
- uses a degradation pre-removal stage plus a restoration stage,
- uses a vision-aided discriminator during adversarial fine-tuning.

The public repository in this workspace keeps the broad SD2 restoration idea, but
the actual codebase is already a research platform with extra deblurring pipelines,
specialized trainers, alternative inference frontends, and monitoring utilities.

## Repository Structure

### 1. Root entrypoints

- `train.py`: main training dispatcher; routes `sd2` to `SD2Trainer` and `cascade_sd2` to `CascadeSD2Trainer`.
- `test.py`: original batch inference entrypoint for the SD2-based HYPIR path.
- `app.py`: Gradio demo frontend that wraps `SD2Enhancer`.
- `inference.py`: simplified batch restoration script for local experiments.
- `inference_moe_hypir.py`: joint MOE-preprocess + HYPIR generation inference path.
- `train_isblur.py`, `train_deblur.py`, `train_expert.py`, `train_moe.py`: local extensions for staged deblurring research.

### 2. Config layer

`configs/` is the experiment switchboard.

- `sd2_train.yaml`: closest to the original paper-style SD2 training path.
- `sd2_finetune_5k.yaml` and `sd2_finetune_5k_improved.yaml`: local fine-tuning variants with altered loss weights and monitoring.
- `sd2_cascade_blur.yaml`: stage-2 cascade training with frozen deblur preprocessing.
- `isblur_pretrain.yaml`, `nafnet_finetune_blur.yaml`, `nafnet_expert.yaml`, `nafnet_moe.yaml`: local deblurring/mixture-of-experts extensions.

### 3. Package layout

`HYPIR/` is split by responsibility:

- `trainer/`: training orchestration
- `enhancer/`: inference-time restoration wrappers
- `model/`: discriminator, blur classifier, NAFNet, MOE, and model wrappers
- `dataset/`: paired data loading, blur labels, expert filtering, synthetic degradation
- `utils/`: instantiation helpers, EMA, captioning, tiled VAE, degradations, optimizer helpers

## Paper-to-Code Mapping

### Paper-native core path

- Restoration backbone:
  `HYPIR/trainer/sd2.py` and `HYPIR/enhancer/sd2.py`
- Core training loop, adversarial loss, EMA, validation:
  `HYPIR/trainer/base.py`
- Vision-aided discriminator:
  `HYPIR/model/D.py`
- Paired restoration dataset and degradation pipeline:
  `HYPIR/dataset/paired.py`, `HYPIR/dataset/realesrgan.py`, `HYPIR/dataset/batch_transform.py`

This path corresponds to the repo's "official" SD2 flow exposed in `README.md`,
`train.py`, `test.py`, and `app.py`.

### Local deblurring extensions

- Blur detection:
  `HYPIR/model/isblur.py`, `train_isblur.py`
- External deblur frontend with NAFNet:
  `HYPIR/model/nafnet.py`, `HYPIR/model/nafnet_wrapper.py`, `train_deblur.py`
- Cascade integration into SD2 training:
  `HYPIR/trainer/cascade_sd2.py`
- Expert specialization and MOE routing:
  `HYPIR/trainer/expert_trainer.py`, `HYPIR/trainer/moe_trainer.py`,
  `HYPIR/model/nafnet_moe.py`, `HYPIR/model/nafnet_moe_wrapper.py`,
  `inference_moe_hypir.py`

## Important Alignment Notes

### 1. The base repo is not a paper-complete reproduction

The paper describes a degradation pre-removal encoder stage before restoration.
In the current repo's base SD2 path, `BaseTrainer.init_vae()` loads the VAE and
immediately freezes it. There is no separate encoder fine-tuning stage in the
mainline SD2 trainer. Your later NAFNet/isBlur cascade effectively acts as an
externalized pre-removal module instead.

### 2. The practical training target is LoRA over SD2 UNet

The paper discusses parameter-efficient tuning, and the repository concretely
implements this through LoRA modules on the SD2 UNet in both training and inference.
The current configs use a larger `lora_rank` than the paper description, which is
a local experiment choice rather than a conceptual rewrite of the project.

### 3. Your modified repo has three layers now

- Layer A: original HYPIR-SD2 restoration path
- Layer B: deblur-specific preprocessing stack (isBlur + NAFNet)
- Layer C: specialization/routing experiments (expert models, MOE, improved configs, SwanLab instrumentation)

Future tasks should state clearly which layer they target. Otherwise it is easy to
compare results across incompatible paths.

## Suggested Mental Model For Future Work

Treat this repository as a two-part system:

1. A generative restoration backend based on SD2 latent restoration.
2. A local deblurring frontend that either preprocesses inputs before HYPIR or
   routes them through specialized restoration branches.

For deblurring research, the most relevant control points are:

- data labeling and filtering: `scripts/add_degradation_label.py`, `HYPIR/dataset/blur_labeled.py`, `HYPIR/dataset/expert.py`
- deblur module design: `HYPIR/model/isblur.py`, `HYPIR/model/nafnet_wrapper.py`, `HYPIR/model/nafnet_moe.py`
- cascade coupling: `HYPIR/trainer/cascade_sd2.py`
- evaluation and inference wiring: `inference.py`, `inference_moe_hypir.py`, `test.py`
- experiment configs: `configs/nafnet_finetune_blur.yaml`, `configs/sd2_cascade_blur.yaml`, `configs/nafnet_moe.yaml`

## Verification Evidence

- Paper-facing repo description: `README.md`
- Original SD2 trainer dispatch: `train.py`
- Main SD2 trainer implementation: `HYPIR/trainer/sd2.py`
- Frozen VAE in base path: `HYPIR/trainer/base.py`
- Cascade deblur path: `HYPIR/trainer/cascade_sd2.py`
- Staged deblur training: `HYPIR/trainer/deblur_trainer.py`
- Expert/MOE extension: `HYPIR/trainer/expert_trainer.py`, `HYPIR/trainer/moe_trainer.py`
- Deblur workflow intent: `CASCADE_TRAINING_GUIDE.md`
