"""Cog runner helpers for the Ideogram 4 trainer."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from dataset_nsfw_classifier import (
    NSFW_VOTE_THRESHOLD,
    build_nsfw_model_specs_csv,
    is_dataset_image,
    is_ignored_dataset_path,
    scan_dataset_directory,
)
from simpletuner.helpers.training.trainer import run_trainer_job

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
INLINE_NSFW_BACKEND_TYPES = "huggingface,csv,aws"
INLINE_NSFW_SAMPLE_TYPES = "image,conditioning"
DEFAULT_TRIGGER_WORD = "TOK"
LOCAL_CAPTION_STRATEGIES = {"textfile", "filename", "instanceprompt"}


class Ideogram4CogRunner:
    """Prepare and launch an Ideogram 4 SimpleTuner job from Cog."""

    def __init__(
        self,
        *,
        base_config_path: Path | str = Path("config") / "ideogram4_train_config.json",
        dataset_root: Path | str = Path("datasets") / "cog",
        output_root: Path | str = Path("output") / "cog",
        config_root: Path | str = Path("config") / "cog",
        debug_log: Path | str = Path("debug.log"),
    ) -> None:
        self.base_config_path = Path(base_config_path)
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.config_root = Path(config_root)
        self.debug_log_path = Path(debug_log)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.config_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        dataset_archive: Path | None,
        hf_dataset: str | None,
        pixel_area: int,
        crop: bool,
        crop_style: str,
        crop_aspect: str,
        trigger_word: str,
        caption_strategy: str,
        train_batch_size: int,
        enable_regional_compile: bool,
        checkpoints_total_limit: int,
        checkpoint_epoch_interval: int,
        checkpoint_step_interval: int,
        max_train_steps: int,
        num_train_epochs: int,
        publishing_config: Optional[Dict[str, Any]],
        hf_token: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stage data, build configs, and launch training."""

        job = job_id or self._new_job_id()
        output_dir = self.output_root / job
        cache_root = self.output_root / f"{job}-cache"
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)

        dataset_config_path, nsfw_scan_report, inline_nsfw_required = self._prepare_dataset_config(
            job_id=job,
            dataset_archive=dataset_archive,
            hf_dataset=hf_dataset,
            pixel_area=pixel_area,
            crop=crop,
            crop_style=crop_style,
            crop_aspect=crop_aspect,
            trigger_word=trigger_word,
            caption_strategy=caption_strategy,
            output_dir=output_dir,
            cache_root=cache_root,
        )

        base_config = self._load_base_config()
        merged_config = dict(base_config)

        pixel_area_value = int(pixel_area)
        merged_config.update(
            {
                "--output_dir": str(output_dir),
                "--cache_dir": str(cache_root / "text"),
                "--data_backend_config": str(dataset_config_path),
                "__job_id__": job,
                "--train_batch_size": int(train_batch_size),
                "--checkpoints_total_limit": max(1, min(int(checkpoints_total_limit), 5)),
                "--checkpoint_epoch_interval": int(checkpoint_epoch_interval),
                "--checkpoint_step_interval": int(checkpoint_step_interval),
                "--max_train_steps": int(max_train_steps),
                "--num_train_epochs": int(num_train_epochs),
                "--resolution": pixel_area_value,
                "--resolution_type": "pixel_area",
                "--minimum_image_size": pixel_area_value,
                "--maximum_image_size": pixel_area_value,
                "--target_downsample_size": pixel_area_value,
                "--validation_resolution": f"{pixel_area_value}x{pixel_area_value}",
            }
        )
        if enable_regional_compile:
            merged_config["--dynamo_backend"] = "inductor"
            merged_config["--dynamo_use_regional_compilation"] = True
        else:
            merged_config.pop("--dynamo_backend", None)
            merged_config.pop("dynamo_backend", None)
            merged_config.pop("--dynamo_use_regional_compilation", None)
            merged_config.pop("dynamo_use_regional_compilation", None)
        trigger = self._normalise_trigger_word(trigger_word)
        merged_config["--validation_prompt"] = self._validation_prompt(trigger)

        merged_config.setdefault("--tracker_project_name", "cog-ideogram4")
        merged_config.setdefault("--tracker_run_name", job)
        merged_config.setdefault("--validation_steps", 50)

        if publishing_config:
            merged_config["publishing_config"] = publishing_config

        if inline_nsfw_required:
            self._force_inline_nsfw_check(merged_config)

        self._apply_hf_token(hf_token)

        training_result = run_trainer_job(merged_config)

        return {
            "job_id": job,
            "output_dir": str(output_dir),
            "dataset_config_path": str(dataset_config_path),
            "nsfw_scan_report_path": (
                None if nsfw_scan_report is None else nsfw_scan_report.get("report_path")
            ),
            "nsfw_scan_summary": (
                None if nsfw_scan_report is None else nsfw_scan_report.get("summary")
            ),
            "training_result": training_result,
        }

    def package_output(self, output_dir: Path, *, target: Path = Path("/tmp/output.zip")) -> Path:
        """Zip the output directory to a single archive for Cog return."""

        output_dir = Path(output_dir)
        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()

        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in output_dir.rglob("*"):
                archive.write(path, arcname=path.relative_to(output_dir))

        return target

    def read_debug_log(self, *, max_bytes: int = 200_000) -> str:
        """Return up to the last `max_bytes` of debug.log."""

        if not self.debug_log_path.exists():
            return ""

        data = self.debug_log_path.read_bytes()
        if len(data) > max_bytes:
            data = data[-max_bytes:]
        return data.decode(errors="replace")

    # Internal helpers ------------------------------------------------- #
    def _prepare_dataset_config(
        self,
        *,
        job_id: str,
        dataset_archive: Path | None,
        hf_dataset: str | None,
        pixel_area: int,
        crop: bool,
        crop_style: str,
        crop_aspect: str,
        trigger_word: str,
        caption_strategy: str,
        output_dir: Path,
        cache_root: Path,
    ) -> tuple[Path, dict[str, Any] | None, bool]:
        if dataset_archive is None and not hf_dataset:
            raise ValueError("Provide either a dataset archive or a Hugging Face dataset name.")
        if dataset_archive is not None and hf_dataset:
            raise ValueError("Choose one data source: dataset archive or Hugging Face dataset, not both.")

        nsfw_scan_report: dict[str, Any] | None = None
        if dataset_archive is not None:
            dataset_dir = self._stage_dataset(dataset_archive, job_id, trigger_word=trigger_word)
            nsfw_scan_report = scan_dataset_directory(
                dataset_dir,
                output_dir / "nsfw_classifier_report.json",
            )
            entry = self._build_local_dataset_entry(
                job_id=job_id,
                dataset_dir=dataset_dir,
                pixel_area=pixel_area,
                crop=crop,
                crop_style=crop_style,
                crop_aspect=crop_aspect,
                trigger_word=trigger_word,
                caption_strategy=caption_strategy,
                output_dir=output_dir,
                cache_root=cache_root,
            )
        else:
            entry = self._build_hf_dataset_entry(
                job_id=job_id,
                dataset_name=hf_dataset or "",
                pixel_area=pixel_area,
                crop=crop,
                crop_style=crop_style,
                crop_aspect=crop_aspect,
                output_dir=output_dir,
                cache_root=cache_root,
            )
        inline_nsfw_required = self._entry_requires_inline_nsfw(entry)

        config_path = self.config_root / f"{job_id}_dataset.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump([entry], handle, indent=2)
        return config_path, nsfw_scan_report, inline_nsfw_required

    @staticmethod
    def _set_cli_arg(config: Dict[str, Any], name: str, value: Any) -> None:
        config.pop(name, None)
        config.pop(f"--{name}", None)
        config[f"--{name}"] = value

    def _force_inline_nsfw_check(self, config: Dict[str, Any]) -> None:
        self._set_cli_arg(config, "enable_nsfw_check", True)
        self._set_cli_arg(config, "nsfw_check_models", build_nsfw_model_specs_csv())
        self._set_cli_arg(config, "nsfw_check_min_votes", NSFW_VOTE_THRESHOLD)
        self._set_cli_arg(config, "nsfw_check_backend_types", INLINE_NSFW_BACKEND_TYPES)
        self._set_cli_arg(config, "nsfw_check_sample_types", INLINE_NSFW_SAMPLE_TYPES)

    @staticmethod
    def _entry_requires_inline_nsfw(entry: Dict[str, Any]) -> bool:
        return str(entry.get("type", "")).lower() in {"aws", "csv", "huggingface"}

    def _build_local_dataset_entry(
        self,
        *,
        job_id: str,
        dataset_dir: Path,
        pixel_area: int,
        crop: bool,
        crop_style: str,
        crop_aspect: str,
        trigger_word: str,
        caption_strategy: str,
        output_dir: Path,
        cache_root: Path,
    ) -> Dict[str, Any]:
        pixel_area_value = int(pixel_area)
        trigger = self._normalise_trigger_word(trigger_word)
        caption_strategy = self._normalise_local_caption_strategy(caption_strategy)
        entry = {
            "id": f"{job_id}-images",
            "type": "local",
            "dataset_type": "image",
            "instance_data_dir": str(dataset_dir),
            "caption_strategy": caption_strategy,
            "metadata_backend": "discovery",
            "crop": bool(crop),
            "crop_style": crop_style,
            "crop_aspect": crop_aspect,
            "minimum_image_size": pixel_area_value,
            "maximum_image_size": pixel_area_value,
            "target_downsample_size": pixel_area_value,
            "resolution": pixel_area_value,
            "resolution_type": "pixel_area",
            "cache_dir_vae": str(cache_root / "vae"),
        }
        if caption_strategy == "instanceprompt":
            entry["instance_prompt"] = self._default_caption(trigger)
        return entry

    def _build_hf_dataset_entry(
        self,
        *,
        job_id: str,
        dataset_name: str,
        pixel_area: int,
        crop: bool,
        crop_style: str,
        crop_aspect: str,
        output_dir: Path,
        cache_root: Path,
    ) -> Dict[str, Any]:
        pixel_area_value = int(pixel_area)
        return {
            "id": f"{job_id}-hf",
            "type": "huggingface",
            "dataset_type": "image",
            "dataset_name": dataset_name,
            "metadata_backend": "huggingface",
            "caption_strategy": "huggingface",
            "crop": bool(crop),
            "crop_style": crop_style,
            "crop_aspect": crop_aspect,
            "minimum_image_size": pixel_area_value,
            "maximum_image_size": pixel_area_value,
            "target_downsample_size": pixel_area_value,
            "resolution": pixel_area_value,
            "resolution_type": "pixel_area",
            "huggingface": {"split": "train"},
            "cache_dir_vae": str(cache_root / "vae"),
        }

    def _stage_dataset(self, archive_path: Path, job_id: str, *, trigger_word: str = DEFAULT_TRIGGER_WORD) -> Path:
        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Dataset archive not found: {archive_path}")

        dest = (self.dataset_root / job_id).resolve()
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)

        if zipfile.is_zipfile(archive_path):
            self._extract_zip(archive_path, dest)
        elif tarfile.is_tarfile(archive_path):
            self._extract_tar(archive_path, dest)
        else:
            raise ValueError("Unsupported dataset archive format. Use .zip or .tar.")

        self._remove_ignored_dataset_files(dest)

        images = [path for path in dest.rglob("*") if is_dataset_image(path)]
        if not images:
            raise ValueError(f"No image files found after extracting {archive_path}.")

        trigger = self._normalise_trigger_word(trigger_word)
        for img in images:
            has_caption = any(
                candidate.exists() for candidate in (img.with_suffix(".txt"), img.with_suffix(".TXT"))
            )
            if not has_caption:
                img.with_suffix(".txt").write_text(
                    self._default_caption(trigger, image_stem=img.stem),
                    encoding="utf-8",
                )

        return dest

    @staticmethod
    def _remove_ignored_dataset_files(dataset_dir: Path) -> None:
        for path in sorted(dataset_dir.rglob("*"), reverse=True):
            if not is_ignored_dataset_path(path):
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    @staticmethod
    def _normalise_trigger_word(trigger_word: str | None) -> str:
        trigger = (trigger_word or DEFAULT_TRIGGER_WORD).strip()
        return trigger or DEFAULT_TRIGGER_WORD

    @staticmethod
    def _normalise_local_caption_strategy(caption_strategy: str | None) -> str:
        strategy = (caption_strategy or "textfile").strip().lower()
        if strategy not in LOCAL_CAPTION_STRATEGIES:
            allowed = ", ".join(sorted(LOCAL_CAPTION_STRATEGIES))
            raise ValueError(f"Unsupported caption_strategy={caption_strategy!r}. Use one of: {allowed}.")
        return strategy

    @classmethod
    def _default_caption(cls, trigger_word: str, *, image_stem: str | None = None) -> str:
        subject = cls._normalise_trigger_word(trigger_word)
        image_hint = f" from image {image_stem}" if image_stem else ""
        return json.dumps(
            {
                "high_level_description": (
                    f"A clean detailed photograph of {subject} as the main subject{image_hint}, "
                    "centered in a square composition with natural proportions and clear identity."
                ),
                "style_description": {
                    "aesthetics": "Clean, realistic, detailed, natural.",
                    "lighting": "Soft even light with clear subject separation.",
                    "photo": "Sharp 35mm photograph, square framing, natural color, high detail.",
                    "medium": "Photograph.",
                    "color_palette": ["#2F3340", "#D8D2C4"],
                },
                "compositional_deconstruction": {
                    "background": "Simple uncluttered background with the subject clearly separated.",
                    "elements": [
                        {
                            "type": "obj",
                            "bbox": [240, 160, 784, 900],
                            "desc": f"{subject} centered as the primary subject, clearly visible and in focus.",
                        }
                    ],
                },
            },
            separators=(",", ":"),
        )

    @classmethod
    def _validation_prompt(cls, trigger_word: str) -> str:
        return cls._default_caption(trigger_word, image_stem=None)

    def _extract_zip(self, archive_path: Path, dest: Path) -> None:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in archive.infolist():
                name = member.filename
                if not name:
                    continue
                if member.is_dir():
                    continue
                path_obj = Path(name)
                if path_obj.is_absolute() or ".." in path_obj.parts:
                    raise ValueError(f"Unsafe zip path detected: {name}")
                target = (dest / path_obj).resolve()
                try:
                    target.relative_to(dest)
                except ValueError:
                    raise ValueError(f"Unsafe zip path detected: {name}")
            archive.extractall(dest)

    def _extract_tar(self, archive_path: Path, dest: Path) -> None:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                name = member.name
                if not name:
                    continue
                if member.isdir():
                    continue
                path_obj = Path(name)
                if path_obj.is_absolute() or ".." in path_obj.parts:
                    raise ValueError(f"Unsafe tar path detected: {name}")
                target = (dest / path_obj).resolve()
                try:
                    target.relative_to(dest)
                except ValueError:
                    raise ValueError(f"Unsafe tar path detected: {name}")
            archive.extractall(dest)

    def _load_base_config(self) -> Dict[str, Any]:
        if not self.base_config_path.exists():
            raise FileNotFoundError(f"Base config not found: {self.base_config_path}")
        with self.base_config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Config at {self.base_config_path} must be a JSON object of CLI args.")
        return data

    def _apply_hf_token(self, hf_token: Optional[str]) -> None:
        if not hf_token:
            return
        for var in ("HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HF_API_TOKEN"):
            os.environ[var] = hf_token

    @staticmethod
    def build_s3_publishing_config(
        *,
        bucket: str,
        job_id: str,
        base_path: str | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        public_base_url: str | None = None,
    ) -> Optional[list[dict[str, Any]]]:
        if not bucket:
            return None

        path_prefix = Ideogram4CogRunner._normalise_s3_base_path(base_path, job_id=job_id)
        entry: dict[str, Any] = {
            "provider": "s3",
            "bucket": bucket,
            "base_path": path_prefix,
        }
        if region:
            entry["region"] = region
        if endpoint_url:
            entry["endpoint_url"] = endpoint_url
        if access_key:
            entry["access_key"] = access_key
        if secret_key:
            entry["secret_key"] = secret_key
        if public_base_url:
            entry["public_base_url"] = public_base_url

        return [entry]

    @staticmethod
    def _normalise_s3_base_path(base_path: str | None, *, job_id: str) -> str:
        path_prefix = (base_path or "cog/ideogram4").strip("/")
        parts = [part for part in path_prefix.split("/") if part]
        if parts and parts[-1] == job_id:
            parts = parts[:-1]
        return "/".join(parts)

    def _new_job_id(self) -> str:
        return f"cog-{uuid.uuid4().hex[:10]}"
