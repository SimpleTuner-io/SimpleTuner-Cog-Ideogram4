# SimpleTuner Cog – Ideogram 4

This repo packages SimpleTuner (via the `SimpleTuner` submodule on the `feature/ideogram-scaled-mm` branch) into a Cog predictor that trains Ideogram 4 LoRAs with a small, opinionated surface.

## What it does
- Accepts either a zip/tar of images or a Hugging Face dataset repo ID. Uploaded archives default to `caption_strategy=textfile`; missing archive `.txt` captions are filled with structured Ideogram JSON using `trigger_word`.
- Runs an NSFW preflight scan on uploaded zip/tar datasets before SimpleTuner starts. The scan uses three standard Hugging Face Transformers classifiers, writes `nsfw_classifier_report.json` into the job output directory, and blocks training when at least two of the three classifiers report NSFW.
- Force-enables SimpleTuner's native VAE-cache NSFW check for remote dataset backends that cannot be scanned ahead of time, including Hugging Face and the same AWS/CSV backend types supported by SimpleTuner configs.
- Auto-generates a `data_backend_config` with your crop settings and pixel-area resolution (applied to `minimum_image_size`, `maximum_image_size`, `target_downsample_size`, and validation resolution).
- Runs the Ideogram 4 FP8 LoRA config in `config/ideogram4_train_config.json` with defaults: batch size slider 1–2 (default 1), rank 32 / alpha 4, `num_train_epochs=25`, `max_train_steps=0`, `checkpoint_epoch_interval=5`, `checkpoint_step_interval=0`, `checkpoints_total_limit=3` (clamped to 5).
- Enables Ideogram's JSON caption wrapping and validation path. Validation uses structured prompts and CFG guidance because short prompts can trigger Ideogram's weaker filtered path.
- Publishes checkpoints/output via `--publishing_config` to an S3-compatible bucket you provide. The job also returns a zip of the `output_dir` for direct download.

## FP8 VRAM
Measured on H100 80GB with native FP8 (`base_model_precision=fp8-torchao`, `quantize_via=pipeline`), rank 32 LoRA, bf16 mixed precision, gradient checkpointing enabled, 1024px square training, and validation disabled:

| Batch size | Peak VRAM |
| --- | ---: |
| 1 | 15,999 MiB / 15.6 GiB |
| 2 | 20,095 MiB / 19.6 GiB |
| 4 | 28,603 MiB / 27.9 GiB |

Validation has a separate generation peak, so keep extra headroom when `ideogram_validation=true`.

## Inputs (Cog)
- `train_data` (optional): zip/tar containing images and optional per-image `.txt` files (same basename). Required if `hf_dataset` is empty.
- `hf_dataset` (optional): Hugging Face dataset repo ID used when `train_data` is not provided. Config mirrors the dreambooth-style HF backend examples.
- `pixel_area`: drives all resolution-related fields.
- `enable_crop`, `crop_style` (`center`/`random`), `crop_aspect` (`square`/`preserve`).
- `trigger_word`: subject token used when generating missing JSON captions and validation prompts.
- `caption_strategy`: caption mode for uploaded archives. `textfile` is the default and uses matching `.txt` files with generated structured JSON fallbacks; `filename` derives captions from filenames; `instanceprompt` uses one structured JSON trigger prompt for every image.
- `train_batch_size` 1–2 (default 1).
- Checkpoint controls: `checkpoints_total_limit` (default 3, max 5), `checkpoint_epoch_interval` (default 5), `checkpoint_step_interval` fixed at 0.
- Training length: `num_train_epochs` (default 25), `max_train_steps` (default 0).
- Publishing (required): `s3_bucket`, plus optional `s3_region`, `s3_endpoint_url`, `s3_base_path`, `s3_public_base_url`, `s3_access_key`, `s3_secret_key`.
- `hf_token` (optional) for gated models/datasets.

## Building/running
```
cog build
cog predict -i train_data=@/path/to/data.zip -i s3_bucket=... -i s3_access_key=... -i s3_secret_key=...
```

## Docs
- Ideogram 4 quickstart: `SimpleTuner/documentation/quickstart/IDEOGRAM4.md`
- Extra background (similar surface): `SimpleTuner/documentation/quickstart/FLUX.md`
