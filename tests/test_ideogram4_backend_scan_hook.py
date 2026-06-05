from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile


def load_backend_module():
    simpletuner = types.ModuleType("simpletuner")
    helpers = types.ModuleType("simpletuner.helpers")
    training = types.ModuleType("simpletuner.helpers.training")
    trainer = types.ModuleType("simpletuner.helpers.training.trainer")
    trainer.run_trainer_job = lambda config: {"config": config}

    sys.modules["simpletuner"] = simpletuner
    sys.modules["simpletuner.helpers"] = helpers
    sys.modules["simpletuner.helpers.training"] = training
    sys.modules["simpletuner.helpers.training.trainer"] = trainer
    sys.modules.pop("ideogram4_backend", None)
    return importlib.import_module("ideogram4_backend")


class Ideogram4BackendScanHookTests(unittest.TestCase):
    def test_prepare_dataset_config_scans_archive_before_writing_config(self) -> None:
        backend = load_backend_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_config = root / "base.json"
            base_config.write_text("{}", encoding="utf-8")
            archive_path = root / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("sample.png", b"not a real image")
                archive.writestr("sample.txt", "caption")

            runner = backend.Ideogram4CogRunner(
                base_config_path=base_config,
                dataset_root=root / "datasets",
                output_root=root / "output",
                config_root=root / "configs",
                debug_log=root / "debug.log",
            )
            config_path = root / "configs" / "job1_dataset.json"
            calls = {}

            def fake_scan(dataset_dir, report_path, **kwargs):
                self.assertFalse(config_path.exists())
                calls["dataset_dir"] = Path(dataset_dir)
                calls["report_path"] = Path(report_path)
                calls["kwargs"] = kwargs
                report = {
                    "report_path": str(report_path),
                    "summary": {"images_scanned": 1, "flagged_images": 0},
                }
                Path(report_path).parent.mkdir(parents=True, exist_ok=True)
                Path(report_path).write_text(json.dumps(report), encoding="utf-8")
                return report

            with patch.object(backend, "scan_dataset_directory", side_effect=fake_scan):
                written_config, report, inline_nsfw_required = runner._prepare_dataset_config(
                    job_id="job1",
                    dataset_archive=archive_path,
                    hf_dataset=None,
                    pixel_area=1024,
                    crop=True,
                    crop_style="center",
                    crop_aspect="square",
                    trigger_word="TOK",
                    caption_strategy="textfile",
                    output_dir=root / "output" / "job1",
                    cache_root=root / "output" / "job1-cache",
                )

            self.assertEqual(written_config, config_path)
            self.assertEqual(report["summary"]["images_scanned"], 1)
            self.assertFalse(inline_nsfw_required)
            self.assertEqual(calls["kwargs"], {})
            self.assertEqual(calls["report_path"].name, "nsfw_classifier_report.json")

            dataset_config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(dataset_config[0]["instance_data_dir"], str(calls["dataset_dir"]))
            self.assertEqual(dataset_config[0]["metadata_backend"], "discovery")
            self.assertEqual(dataset_config[0]["caption_strategy"], "textfile")
            self.assertNotIn("instance_prompt", dataset_config[0])

    def test_remote_dataset_forces_inline_nsfw_check_in_trainer_config(self) -> None:
        backend = load_backend_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_config = root / "base.json"
            base_config.write_text("{}", encoding="utf-8")
            runner = backend.Ideogram4CogRunner(
                base_config_path=base_config,
                dataset_root=root / "datasets",
                output_root=root / "output",
                config_root=root / "configs",
                debug_log=root / "debug.log",
            )

            result = runner.run(
                dataset_archive=None,
                hf_dataset="user/dataset",
                pixel_area=1024,
                crop=True,
                crop_style="center",
                crop_aspect="square",
                trigger_word="TOK",
                caption_strategy="textfile",
                train_batch_size=1,
                checkpoints_total_limit=1,
                checkpoint_epoch_interval=1,
                checkpoint_step_interval=0,
                max_train_steps=1,
                num_train_epochs=1,
                publishing_config=None,
                job_id="job2",
            )

            trainer_config = result["training_result"]["config"]
            self.assertIs(trainer_config["--enable_nsfw_check"], True)
            self.assertEqual(trainer_config["--cache_dir"], str(root / "output" / "job2-cache" / "text"))
            self.assertEqual(trainer_config["--nsfw_check_min_votes"], 2)
            self.assertEqual(trainer_config["--nsfw_check_backend_types"], "huggingface,csv,aws")
            self.assertEqual(trainer_config["--nsfw_check_sample_types"], "image,conditioning")
            nsfw_models = trainer_config["--nsfw_check_models"]
            self.assertRegex(nsfw_models, r"(Falconsai/nsfw_image_detection|hf_falconsai)")
            self.assertRegex(nsfw_models, r"(AdamCodd/vit-base-nsfw-detector|hf_adamcodd)")
            self.assertRegex(nsfw_models, r"(hoangtrung1801/nsfw-vit-model|hf_hoangtrung)")

    def test_inline_nsfw_is_required_for_remote_backend_types_only(self) -> None:
        backend = load_backend_module()

        self.assertTrue(backend.Ideogram4CogRunner._entry_requires_inline_nsfw({"type": "aws"}))
        self.assertTrue(backend.Ideogram4CogRunner._entry_requires_inline_nsfw({"type": "csv"}))
        self.assertTrue(backend.Ideogram4CogRunner._entry_requires_inline_nsfw({"type": "huggingface"}))
        self.assertFalse(backend.Ideogram4CogRunner._entry_requires_inline_nsfw({"type": "local"}))

    def test_archive_without_captions_gets_structured_ideogram_caption(self) -> None:
        backend = load_backend_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_config = root / "base.json"
            base_config.write_text("{}", encoding="utf-8")
            archive_path = root / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("sample.png", b"not a real image")

            runner = backend.Ideogram4CogRunner(
                base_config_path=base_config,
                dataset_root=root / "datasets",
                output_root=root / "output",
                config_root=root / "configs",
                debug_log=root / "debug.log",
            )
            dataset_dir = runner._stage_dataset(archive_path, "job3", trigger_word="my_subject")
            caption_path = dataset_dir / "sample.txt"
            self.assertTrue(caption_path.exists())

            caption = json.loads(caption_path.read_text(encoding="utf-8"))
            self.assertIn("my_subject", caption["high_level_description"])
            self.assertIn("color_palette", caption["style_description"])
            self.assertEqual(caption["compositional_deconstruction"]["elements"][0]["type"], "obj")

    def test_archive_caption_generation_handles_dotted_filenames(self) -> None:
        backend = load_backend_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_config = root / "base.json"
            base_config.write_text("{}", encoding="utf-8")
            archive_path = root / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("subject.with.extra.dots.webp", b"not a real image")

            runner = backend.Ideogram4CogRunner(
                base_config_path=base_config,
                dataset_root=root / "datasets",
                output_root=root / "output",
                config_root=root / "configs",
                debug_log=root / "debug.log",
            )
            dataset_dir = runner._stage_dataset(archive_path, "job6", trigger_word="TOK")

            self.assertTrue((dataset_dir / "subject.with.extra.dots.txt").exists())
            self.assertFalse((dataset_dir / "subject.with.extra.txt").exists())

    def test_local_dataset_entry_uses_default_trigger_json_instance_prompt(self) -> None:
        backend = load_backend_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_config = root / "base.json"
            base_config.write_text("{}", encoding="utf-8")
            runner = backend.Ideogram4CogRunner(
                base_config_path=base_config,
                dataset_root=root / "datasets",
                output_root=root / "output",
                config_root=root / "configs",
                debug_log=root / "debug.log",
            )

            entry = runner._build_local_dataset_entry(
                job_id="job5",
                dataset_dir=root / "datasets" / "job5",
                pixel_area=1024,
                crop=True,
                crop_style="center",
                crop_aspect="square",
                trigger_word="",
                caption_strategy="instanceprompt",
                output_dir=root / "output" / "job5",
                cache_root=root / "output" / "job5-cache",
            )

            self.assertEqual(entry["caption_strategy"], "instanceprompt")
            self.assertEqual(entry["cache_dir_vae"], str(root / "output" / "job5-cache" / "vae"))
            prompt = json.loads(entry["instance_prompt"])
            self.assertIn("TOK", prompt["high_level_description"])
            self.assertIn("color_palette", prompt["style_description"])

    def test_local_dataset_entry_allows_filename_caption_strategy(self) -> None:
        backend = load_backend_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_config = root / "base.json"
            base_config.write_text("{}", encoding="utf-8")
            runner = backend.Ideogram4CogRunner(
                base_config_path=base_config,
                dataset_root=root / "datasets",
                output_root=root / "output",
                config_root=root / "configs",
                debug_log=root / "debug.log",
            )

            entry = runner._build_local_dataset_entry(
                job_id="job7",
                dataset_dir=root / "datasets" / "job7",
                pixel_area=1024,
                crop=True,
                crop_style="center",
                crop_aspect="square",
                trigger_word="TOK",
                caption_strategy="filename",
                output_dir=root / "output" / "job7",
                cache_root=root / "output" / "job7-cache",
            )

            self.assertEqual(entry["caption_strategy"], "filename")
            self.assertNotIn("instance_prompt", entry)

    def test_stage_dataset_removes_macos_resource_fork_files(self) -> None:
        backend = load_backend_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_config = root / "base.json"
            base_config.write_text("{}", encoding="utf-8")
            archive_path = root / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("domokun/sample.jpg", b"not a real image")
                archive.writestr("domokun/._sample.jpg", b"appledouble metadata")
                archive.writestr("__MACOSX/domokun/._sample.jpg", b"appledouble metadata")
                archive.writestr("domokun/.DS_Store", b"metadata")

            runner = backend.Ideogram4CogRunner(
                base_config_path=base_config,
                dataset_root=root / "datasets",
                output_root=root / "output",
                config_root=root / "configs",
                debug_log=root / "debug.log",
            )
            dataset_dir = runner._stage_dataset(archive_path, "job4", trigger_word="TOK")

            self.assertTrue((dataset_dir / "domokun" / "sample.jpg").exists())
            self.assertTrue((dataset_dir / "domokun" / "sample.txt").exists())
            self.assertFalse((dataset_dir / "domokun" / "._sample.jpg").exists())
            self.assertFalse((dataset_dir / "__MACOSX").exists())
            self.assertFalse((dataset_dir / "domokun" / ".DS_Store").exists())

    def test_s3_base_path_defaults_to_parent_prefix(self) -> None:
        backend = load_backend_module()

        config = backend.Ideogram4CogRunner.build_s3_publishing_config(
            bucket="bucket",
            job_id="cog-abc123",
        )

        self.assertEqual(config[0]["base_path"], "cog/ideogram4")

    def test_s3_base_path_strips_trailing_current_job_id(self) -> None:
        backend = load_backend_module()

        config = backend.Ideogram4CogRunner.build_s3_publishing_config(
            bucket="bucket",
            job_id="cog-abc123",
            base_path="training/cog/ideogram4/cog-abc123",
        )

        self.assertEqual(config[0]["base_path"], "training/cog/ideogram4")

    def test_s3_base_path_keeps_parent_prefix(self) -> None:
        backend = load_backend_module()

        config = backend.Ideogram4CogRunner.build_s3_publishing_config(
            bucket="bucket",
            job_id="cog-abc123",
            base_path="/training/cog/ideogram4/",
        )

        self.assertEqual(config[0]["base_path"], "training/cog/ideogram4")


if __name__ == "__main__":
    unittest.main()
