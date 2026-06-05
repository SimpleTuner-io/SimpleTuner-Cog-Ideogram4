from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

import dataset_nsfw_classifier as nsfw


class DatasetNsfwClassifierTests(unittest.TestCase):
    def test_score_sum_maps_common_labels(self) -> None:
        scores = [
            {"label": "neutral", "score": 0.25},
            {"label": "porn", "score": 0.45},
            {"label": "sexy", "score": 0.20},
            {"label": "unknown", "score": 0.10},
        ]

        self.assertEqual(nsfw._score_sum(scores, nsfw.NSFW_LABEL_HINTS), 0.65)
        self.assertEqual(nsfw._score_sum(scores, nsfw.SFW_LABEL_HINTS), 0.25)

    def test_dataset_report_flags_two_of_three_nsfw_images(self) -> None:
        dataset_dir = Path("/tmp/dataset")
        report = nsfw._build_dataset_report(
            dataset_dir=dataset_dir,
            report_path=Path("/tmp/report.json"),
            images=[dataset_dir / "one.png"],
            image_results=[
                {
                    "image": str(dataset_dir / "one.png"),
                    "relative_path": "one.png",
                    "classifiers": [
                        {"key": "hf_falconsai", "verdict": "nsfw", "nsfw_score": 0.9},
                        {"key": "hf_adamcodd", "verdict": "sfw", "nsfw_score": 0.1},
                        {"key": "hf_marqo", "verdict": "nsfw", "nsfw_score": 0.7},
                    ],
                    "summary": {
                        "known_verdicts": 3,
                        "nsfw_votes": 2,
                        "sfw_votes": 1,
                        "majority_verdict": "nsfw",
                    },
                }
            ],
            started_at=time.perf_counter(),
        )

        self.assertEqual(report["summary"]["flagged_images"], 1)
        self.assertEqual(report["vote_threshold"], 2)
        self.assertEqual(report["flagged_images"][0]["relative_path"], "one.png")
        self.assertEqual(report["summary"]["classifier_verdicts"]["hf_adamcodd"]["sfw"], 1)

    def test_dataset_report_does_not_flag_one_of_three_nsfw_images(self) -> None:
        dataset_dir = Path("/tmp/dataset")
        report = nsfw._build_dataset_report(
            dataset_dir=dataset_dir,
            report_path=Path("/tmp/report.json"),
            images=[dataset_dir / "one.png"],
            image_results=[
                {
                    "image": str(dataset_dir / "one.png"),
                    "relative_path": "one.png",
                    "classifiers": [
                        {"key": "hf_falconsai", "verdict": "nsfw", "nsfw_score": 0.9},
                        {"key": "hf_adamcodd", "verdict": "sfw", "nsfw_score": 0.1},
                        {"key": "hf_marqo", "verdict": "sfw", "nsfw_score": 0.2},
                    ],
                    "summary": {
                        "known_verdicts": 3,
                        "nsfw_votes": 1,
                        "sfw_votes": 2,
                        "majority_verdict": "sfw",
                    },
                }
            ],
            started_at=time.perf_counter(),
        )

        self.assertEqual(report["summary"]["flagged_images"], 0)

    def test_dataset_image_filter_skips_macos_metadata(self) -> None:
        self.assertTrue(nsfw.is_dataset_image(Path("domokun/sample.webp")))
        self.assertFalse(nsfw.is_dataset_image(Path("domokun/._sample.webp")))
        self.assertFalse(nsfw.is_dataset_image(Path("__MACOSX/domokun/._sample.webp")))
        self.assertFalse(nsfw.is_dataset_image(Path("domokun/.DS_Store")))

    def test_ensure_baked_model_image_processors_creates_missing_vit_processor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "hf_hoangtrung"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "vit", "image_size": 224}),
                encoding="utf-8",
            )
            (model_dir / "model.safetensors").write_text("weights", encoding="utf-8")

            with patch.dict(os.environ, {"NSFW_CLASSIFIER_MODEL_DIR": str(root)}):
                nsfw.ensure_baked_model_image_processors()

            processor = json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
            self.assertEqual(processor["image_processor_type"], "ViTImageProcessor")
            self.assertEqual(processor["size"], {"height": 224, "width": 224})

    def test_classifier_error_is_unknown_not_image_error_when_other_classifiers_succeed(self) -> None:
        report = nsfw._build_dataset_report(
            dataset_dir=Path("/tmp/dataset"),
            report_path=Path("/tmp/report.json"),
            images=[Path("/tmp/dataset/one.png")],
            image_results=[
                {
                    "image": "/tmp/dataset/one.png",
                    "relative_path": "one.png",
                    "classifiers": [
                        {"key": "hf_falconsai", "verdict": "sfw"},
                        {"key": "hf_adamcodd", "verdict": "sfw"},
                        {"key": "hf_hoangtrung", "verdict": "unknown", "error": "missing processor"},
                    ],
                    "summary": {
                        "known_verdicts": 2,
                        "nsfw_votes": 0,
                        "sfw_votes": 2,
                        "majority_verdict": "sfw",
                    },
                }
            ],
            started_at=time.perf_counter(),
        )

        self.assertEqual(report["summary"]["error_images"], 0)
        self.assertEqual(report["summary"]["images_scanned"], 1)


if __name__ == "__main__":
    unittest.main()
