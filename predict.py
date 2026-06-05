"""Cog predictor for Ideogram 4 LoRA training."""

from __future__ import annotations

import ctypes
import os
import site
from pathlib import Path as FilePath
from typing import Optional

from cog import BasePredictor, Input, Path, Secret


def _secret_value(secret: Optional[Secret | str]) -> Optional[str]:
    if secret is None:
        return None

    get_secret_value = getattr(secret, "get_secret_value", None)
    if callable(get_secret_value):
        return get_secret_value()

    return str(secret)


def _prefer_cuda_python_wheel_libraries() -> None:
    site_packages = site.getsitepackages()[0]
    library_paths = [
        os.path.join(site_packages, "nvidia", "nccl", "lib"),
        os.path.join(site_packages, "nvidia", "cusparselt", "lib"),
    ]
    existing_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    preferred_paths = [path for path in library_paths if os.path.isdir(path)]
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        preferred_paths + [path for path in existing_paths if path and path not in preferred_paths]
    )

    nccl_path = os.path.join(site_packages, "nvidia", "nccl", "lib", "libnccl.so.2")
    if os.path.exists(nccl_path):
        ctypes.CDLL(nccl_path, mode=ctypes.RTLD_GLOBAL)


_prefer_cuda_python_wheel_libraries()

from ideogram4_backend import Ideogram4CogRunner


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.runner = Ideogram4CogRunner()

    def predict(
        self,
        train_data: Optional[Path] = Input(
            description="Zip or tar of training images. Matching .txt captions are used when present; missing captions get structured Ideogram JSON using trigger_word.",
            default=None,
        ),
        hf_dataset: Optional[str] = Input(
            description="Hugging Face dataset repo ID to use when no archive is provided (e.g., user/dataset).",
            default=None,
        ),
        pixel_area: int = Input(
            description="Pixel-area resolution; also applied to min/max/target size and validation resolution.",
            default=1024,
            ge=256,
            le=2048,
        ),
        enable_crop: bool = Input(
            description="Enable cropping when preparing samples.",
            default=True,
        ),
        crop_style: str = Input(
            description="Crop location style when cropping is enabled.",
            choices=["center", "random"],
            default="center",
        ),
        crop_aspect: str = Input(
            description="Aspect handling for crops: square (1:1) or preserve source aspect.",
            choices=["square", "preserve"],
            default="square",
        ),
        trigger_word: str = Input(
            description="Subject token used in generated JSON captions and validation prompts.",
            default="TOK",
        ),
        caption_strategy: str = Input(
            description="Caption strategy for uploaded archives. textfile uses matching .txt captions and generated JSON fallbacks; filename derives captions from filenames; instanceprompt uses one structured JSON trigger prompt for every image.",
            choices=["textfile", "filename", "instanceprompt"],
            default="textfile",
        ),
        train_batch_size: int = Input(
            description="Training batch size (1-2). Keep 1 unless the GPU has enough memory.",
            default=1,
            ge=1,
            le=2,
        ),
        enable_regional_compile: bool = Input(
            description="Enable torch.compile with regional compilation for the Ideogram transformer. Recommended for native FP8; disable only if compile causes a runtime issue.",
            default=True,
        ),
        checkpoints_total_limit: int = Input(
            description="Maximum checkpoints to retain (max 5).",
            default=3,
            ge=1,
            le=5,
        ),
        checkpoint_epoch_interval: int = Input(
            description="Write a checkpoint every N epochs.",
            default=5,
            ge=1,
        ),
        num_train_epochs: int = Input(
            description="Total training epochs.",
            default=25,
            ge=1,
        ),
        max_train_steps: int = Input(
            description="Optional max train steps override (0 disables the step limit).",
            default=0,
            ge=0,
        ),
        s3_bucket: str = Input(
            description="S3-compatible bucket for publishing checkpoints.",
            default="",
        ),
        s3_region: Optional[str] = Input(
            description="S3 region (optional).",
            default=None,
        ),
        s3_endpoint_url: Optional[str] = Input(
            description="Custom S3 endpoint URL (for non-AWS providers).",
            default=None,
        ),
        s3_base_path: Optional[str] = Input(
            description="Parent prefix inside the bucket. The job id is appended by the publisher (defaults to cog/ideogram4).",
            default=None,
        ),
        s3_public_base_url: Optional[str] = Input(
            description="Public base URL to build shareable links (optional).",
            default=None,
        ),
        s3_access_key: Optional[Secret] = Input(
            description="S3 access key (leave blank to use IAM/instance roles).",
            default=None,
        ),
        s3_secret_key: Optional[Secret] = Input(
            description="S3 secret key (leave blank to use IAM/instance roles).",
            default=None,
        ),
        hf_token: Optional[Secret] = Input(
            description="Hugging Face token for model or dataset access.",
            default=None,
        ),
        return_logs: bool = Input(
            description="Print the tail of debug.log.",
            default=True,
        ),
    ) -> Path:
        job_id = self.runner._new_job_id()

        if not s3_bucket:
            raise ValueError("s3_bucket is required to publish checkpoints.")

        dataset_path = FilePath(train_data) if train_data else None
        hf_dataset_value = hf_dataset or None

        hf_token_value = _secret_value(hf_token)
        s3_access_value = _secret_value(s3_access_key)
        s3_secret_value = _secret_value(s3_secret_key)

        publishing_config = self.runner.build_s3_publishing_config(
            bucket=s3_bucket,
            job_id=job_id,
            base_path=s3_base_path,
            region=s3_region,
            endpoint_url=s3_endpoint_url,
            access_key=s3_access_value,
            secret_key=s3_secret_value,
            public_base_url=s3_public_base_url,
        )

        run_result = self.runner.run(
            dataset_archive=dataset_path,
            hf_dataset=hf_dataset_value,
            pixel_area=pixel_area,
            crop=enable_crop,
            crop_style=crop_style,
            crop_aspect=crop_aspect,
            trigger_word=trigger_word,
            caption_strategy=caption_strategy,
            train_batch_size=train_batch_size,
            enable_regional_compile=enable_regional_compile,
            checkpoints_total_limit=checkpoints_total_limit,
            checkpoint_epoch_interval=checkpoint_epoch_interval,
            checkpoint_step_interval=0,
            max_train_steps=max_train_steps,
            num_train_epochs=num_train_epochs,
            publishing_config=publishing_config,
            hf_token=hf_token_value,
            job_id=job_id,
        )

        nsfw_summary = run_result.get("nsfw_scan_summary")
        nsfw_report_path = run_result.get("nsfw_scan_report_path")
        if nsfw_summary and nsfw_report_path:
            print(
                "\nNSFW preflight scan completed: "
                f"{nsfw_summary.get('images_scanned', 0)} image(s) scanned, "
                f"{nsfw_summary.get('flagged_images', 0)} flagged. "
                f"Report: {nsfw_report_path}"
            )

        archive_path = self.runner.package_output(
            FilePath(run_result["output_dir"]),
            target=FilePath("/tmp") / f"{job_id}_output.zip",
        )

        if return_logs:
            log_tail = self.runner.read_debug_log()
            if log_tail:
                print("\n=== debug.log tail ===")
                print(log_tail[-5000:])

        if publishing_config:
            path_prefix = publishing_config[0].get("base_path", "").lstrip("/")
            remote_prefix = "/".join(part for part in (path_prefix, job_id) if part)
            if s3_public_base_url:
                remote_hint = f"{s3_public_base_url.rstrip('/')}/{remote_prefix}"
            elif s3_endpoint_url:
                remote_hint = f"{s3_endpoint_url.rstrip('/')}/{s3_bucket}/{remote_prefix}"
            else:
                remote_hint = f"s3://{s3_bucket}/{remote_prefix}"
            print(f"\nPublished checkpoints will be stored under: {remote_hint}")

        return Path(archive_path)
