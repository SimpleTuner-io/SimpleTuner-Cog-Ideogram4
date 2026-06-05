"""HF-only NSFW preflight scan for staged image datasets."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import threading
import time
from typing import Any


HF_MODELS = {
    "hf_falconsai": {
        "model_id": "Falconsai/nsfw_image_detection",
        "display_name": "FalconsAI NSFW Image Detection",
    },
    "hf_adamcodd": {
        "model_id": "AdamCodd/vit-base-nsfw-detector",
        "display_name": "AdamCodd ViT NSFW Detector",
    },
    "hf_hoangtrung": {
        "model_id": "hoangtrung1801/nsfw-vit-model",
        "display_name": "HoangTrung ViT NSFW Model",
    },
}
DEFAULT_BAKED_ROOT = "/opt/nsfw-classifier-comparison"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
IGNORED_DATASET_NAMES = {".DS_Store", "Icon\r"}
NSFW_LABEL_HINTS = ("nsfw", "unsafe", "porn", "hentai", "sexy", "explicit", "adult")
SFW_LABEL_HINTS = ("sfw", "safe", "neutral", "normal", "drawing")
NSFW_SCORE_THRESHOLD = 0.5
NSFW_VOTE_THRESHOLD = 2
TOP_K = 5


class DatasetSafetyError(ValueError):
    """Raised when a dataset preflight scan finds unsafe images."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def _normalize_label(label: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in label).strip("_")


def _label_matches(label: str, hints: tuple[str, ...]) -> bool:
    normalized = _normalize_label(label)
    parts = {part for part in normalized.split("_") if part}
    return any(hint in normalized or hint in parts for hint in hints)


def _score_sum(scores: list[dict[str, Any]], hints: tuple[str, ...]) -> float | None:
    matched = [
        float(item["score"])
        for item in scores
        if _label_matches(str(item["label"]), hints)
    ]
    if not matched:
        return None
    return float(sum(matched))


def _verdict(nsfw_score: float | None) -> str:
    if nsfw_score is None:
        return "unknown"
    return "nsfw" if nsfw_score >= NSFW_SCORE_THRESHOLD else "sfw"


def _round_scores(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": str(item["label"]),
            "score": round(float(item["score"]), 6),
        }
        for item in scores
    ]


def _has_hf_model_files(model_dir: Path) -> bool:
    return (model_dir / "config.json").is_file() and (
        any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
    )


def _resolve_hf_model_path(model_key: str, model_id: str) -> str:
    baked_root = Path(os.environ.get("NSFW_CLASSIFIER_MODEL_DIR", DEFAULT_BAKED_ROOT))
    baked_dir = baked_root / model_key
    if _has_hf_model_files(baked_dir):
        return str(baked_dir)
    return model_id


def _default_vit_image_processor_config(model_dir: Path) -> dict[str, Any]:
    image_size = 224
    config_path = model_dir / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            image_size = int(config.get("image_size", image_size))
        except Exception:
            image_size = 224

    return {
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_processor_type": "ViTImageProcessor",
        "image_std": [0.5, 0.5, 0.5],
        "resample": 2,
        "rescale_factor": 0.00392156862745098,
        "size": {"height": image_size, "width": image_size},
    }


def ensure_baked_model_image_processors() -> None:
    """Add missing image processor metadata to baked ViT classifier snapshots."""

    for model_key, info in HF_MODELS.items():
        model_path = Path(_resolve_hf_model_path(model_key, info["model_id"]))
        if not model_path.is_dir():
            continue

        processor_path = model_path / "preprocessor_config.json"
        if processor_path.is_file():
            continue

        config_path = model_path / "config.json"
        if not config_path.is_file():
            continue

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if config.get("model_type") != "vit":
            continue

        processor_path.write_text(
            json.dumps(_default_vit_image_processor_config(model_path), indent=2),
            encoding="utf-8",
        )


def build_nsfw_model_specs_csv() -> str:
    """Return model specs for SimpleTuner's native inline NSFW checker."""

    ensure_baked_model_image_processors()
    return ",".join(
        f"{_resolve_hf_model_path(model_key, info['model_id'])}:threshold={NSFW_SCORE_THRESHOLD}"
        for model_key, info in HF_MODELS.items()
    )


def _coerce_scores(raw_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": str(item.get("label", "")),
            "score": float(item.get("score", 0.0)),
        }
        for item in raw_scores
    ]


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def is_ignored_dataset_path(path: Path) -> bool:
    parts = Path(path).parts
    return any(part == "__MACOSX" or part.startswith("._") or part in IGNORED_DATASET_NAMES for part in parts)


def is_dataset_image(path: Path) -> bool:
    path = Path(path)
    return path.suffix.lower() in IMAGE_EXTS and not is_ignored_dataset_path(path)


class NsfwClassifierRunner:
    """Run the fixed Hugging Face classifier set against local files."""

    def __init__(self) -> None:
        self._hf_pipelines: dict[str, Any] = {}
        self._hf_load_errors: dict[str, str] = {}
        self._hf_lock = threading.Lock()

    def close(self) -> None:
        self._hf_pipelines.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_hf_pipeline(
        self,
        *,
        model_key: str,
        model_id: str,
    ) -> Any:
        with self._hf_lock:
            if model_key in self._hf_pipelines:
                return self._hf_pipelines[model_key]
            if model_key in self._hf_load_errors:
                raise RuntimeError(self._hf_load_errors[model_key])

            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            ensure_baked_model_image_processors()
            model_path = _resolve_hf_model_path(model_key, model_id)
            try:
                classifier = pipeline(
                    "image-classification",
                    model=model_path,
                    device=device,
                    trust_remote_code=False,
                )
            except Exception as exc:
                self._hf_load_errors[model_key] = str(exc)
                raise
            self._hf_pipelines[model_key] = classifier
            return classifier

    def _run_hf_classifier(
        self,
        *,
        image: Path,
        model_key: str,
        model_id: str,
        display_name: str,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        classifier = self._load_hf_pipeline(
            model_key=model_key,
            model_id=model_id,
        )
        raw_output = classifier(str(image), top_k=None)
        scores = _coerce_scores(raw_output)
        scores.sort(key=lambda item: item["score"], reverse=True)
        nsfw_score = _score_sum(scores, NSFW_LABEL_HINTS)
        sfw_score = _score_sum(scores, SFW_LABEL_HINTS)
        top = scores[0] if scores else {"label": None, "score": None}
        return {
            "key": model_key,
            "backend": "transformers",
            "model_id": model_id,
            "display_name": display_name,
            "top_label": top["label"],
            "top_score": None if top["score"] is None else round(float(top["score"]), 6),
            "nsfw_score": None if nsfw_score is None else round(nsfw_score, 6),
            "sfw_score": None if sfw_score is None else round(sfw_score, 6),
            "verdict": _verdict(nsfw_score),
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
            "scores": _round_scores(scores[:TOP_K]),
        }

    def classify_image(self, image: Path) -> dict[str, Any]:
        image_path = Path(image)
        results = [
            self._safe_run_hf_classifier(image_path, model_key, info)
            for model_key, info in HF_MODELS.items()
        ]
        nsfw_votes = sum(1 for result in results if result["verdict"] == "nsfw")
        known_votes = sum(1 for result in results if result["verdict"] != "unknown")
        classifier_errors = [result for result in results if result.get("error")]
        return {
            "image": str(image_path),
            "threshold": NSFW_SCORE_THRESHOLD,
            "vote_threshold": NSFW_VOTE_THRESHOLD,
            "classifiers": results,
            "summary": {
                "count": len(results),
                "known_verdicts": known_votes,
                "nsfw_votes": nsfw_votes,
                "sfw_votes": known_votes - nsfw_votes,
                "majority_verdict": "nsfw" if nsfw_votes >= NSFW_VOTE_THRESHOLD else "sfw",
            },
            **({"error": "; ".join(str(result["error"]) for result in classifier_errors)} if not known_votes else {}),
        }

    def _safe_run_hf_classifier(
        self,
        image_path: Path,
        model_key: str,
        info: dict[str, str],
    ) -> dict[str, Any]:
        try:
            return self._run_hf_classifier(
                image=image_path,
                model_key=model_key,
                model_id=info["model_id"],
                display_name=info["display_name"],
            )
        except Exception as exc:
            return {
                "key": model_key,
                "backend": "transformers",
                "model_id": info["model_id"],
                "display_name": info["display_name"],
                "top_label": None,
                "top_score": None,
                "nsfw_score": None,
                "sfw_score": None,
                "verdict": "unknown",
                "elapsed_ms": 0.0,
                "scores": [],
                "error": str(exc),
            }


def _image_flag_summary(image_result: dict[str, Any]) -> dict[str, Any]:
    classifiers = image_result.get("classifiers", [])
    nsfw_scores = [
        float(result["nsfw_score"])
        for result in classifiers
        if result.get("nsfw_score") is not None
    ]
    return {
        "image": image_result.get("image"),
        "relative_path": image_result.get("relative_path"),
        "majority_verdict": image_result.get("summary", {}).get("majority_verdict"),
        "nsfw_votes": image_result.get("summary", {}).get("nsfw_votes", 0),
        "known_verdicts": image_result.get("summary", {}).get("known_verdicts", 0),
        "max_nsfw_score": round(max(nsfw_scores), 6) if nsfw_scores else None,
    }


def _build_dataset_report(
    *,
    dataset_dir: Path,
    report_path: Path,
    images: list[Path],
    image_results: list[dict[str, Any]],
    started_at: float,
) -> dict[str, Any]:
    classifier_verdicts = {
        key: {"nsfw": 0, "sfw": 0, "unknown": 0}
        for key in HF_MODELS
    }
    flagged = []
    errors = []

    for image_result in image_results:
        if image_result.get("error"):
            errors.append(
                {
                    "image": image_result.get("image"),
                    "relative_path": image_result.get("relative_path"),
                    "error": image_result.get("error"),
                }
            )
            continue

        for result in image_result.get("classifiers", []):
            key = result.get("key")
            verdict = result.get("verdict", "unknown")
            if key in classifier_verdicts and verdict in classifier_verdicts[key]:
                classifier_verdicts[key][verdict] += 1

        if image_result.get("summary", {}).get("nsfw_votes", 0) >= NSFW_VOTE_THRESHOLD:
            flagged.append(_image_flag_summary(image_result))

    return {
        "dataset_dir": str(dataset_dir),
        "report_path": str(report_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "threshold": NSFW_SCORE_THRESHOLD,
        "vote_threshold": NSFW_VOTE_THRESHOLD,
        "models": [
            {"key": key, **info}
            for key, info in HF_MODELS.items()
        ],
        "summary": {
            "images_found": len(images),
            "images_scanned": len(image_results) - len(errors),
            "flagged_images": len(flagged),
            "error_images": len(errors),
            "classifier_verdicts": classifier_verdicts,
        },
        "flagged_images": flagged,
        "errors": errors,
        "images": image_results,
    }


def _format_safety_error(report: dict[str, Any]) -> str:
    flagged = report.get("flagged_images", [])
    preview = ", ".join(str(item.get("relative_path") or item.get("image")) for item in flagged[:10])
    remaining = len(flagged) - min(len(flagged), 10)
    if remaining:
        preview = f"{preview}, and {remaining} more"
    return (
        f"NSFW dataset preflight scan flagged {len(flagged)} image(s) with "
        f"at least {NSFW_VOTE_THRESHOLD}/{len(HF_MODELS)} HF classifiers; "
        "training was not started. "
        f"Report: {report.get('report_path')}. "
        f"Flagged files: {preview}"
    )


def scan_dataset_directory(dataset_dir: Path, report_path: Path) -> dict[str, Any]:
    """Classify every image in a staged local dataset and write a JSON report."""

    dataset_dir = Path(dataset_dir).resolve()
    report_path = Path(report_path)
    images = sorted(path for path in dataset_dir.rglob("*") if is_dataset_image(path))
    if not images:
        raise ValueError(f"No image files found in dataset directory: {dataset_dir}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    image_results: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    runner = NsfwClassifierRunner()
    try:
        for index, image in enumerate(images, start=1):
            relative_path = _safe_relative(image, dataset_dir)
            print(f"NSFW preflight scan {index}/{len(images)}: {relative_path}")
            try:
                image_result = runner.classify_image(image)
                image_result["relative_path"] = relative_path
            except Exception as exc:
                image_result = {
                    "image": str(image),
                    "relative_path": relative_path,
                    "threshold": NSFW_SCORE_THRESHOLD,
                    "vote_threshold": NSFW_VOTE_THRESHOLD,
                    "classifiers": [],
                    "summary": {
                        "count": 0,
                        "known_verdicts": 0,
                        "nsfw_votes": 0,
                        "sfw_votes": 0,
                        "majority_verdict": "unknown",
                    },
                    "error": str(exc),
                }
            image_results.append(image_result)
    finally:
        runner.close()

    report = _build_dataset_report(
        dataset_dir=dataset_dir,
        report_path=report_path,
        images=images,
        image_results=image_results,
        started_at=started_at,
    )
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    error_count = int(report["summary"]["error_images"])
    if error_count:
        raise RuntimeError(
            f"NSFW dataset preflight scan failed on {error_count} image(s); "
            f"training was not started. Report: {report_path}"
        )

    if report["summary"]["flagged_images"]:
        raise DatasetSafetyError(_format_safety_error(report), report)

    return report
