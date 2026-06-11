from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest


def load_predict_module():
    cog = types.ModuleType("cog")

    class BasePredictor:
        pass

    class Secret:
        def __init__(self, value: str):
            self.value = value

        def get_secret_value(self) -> str:
            return self.value

    def Input(*, default=None, **kwargs):
        return default

    cog.BasePredictor = BasePredictor
    cog.Input = Input
    cog.Path = Path
    cog.Secret = Secret
    sys.modules["cog"] = cog

    backend = types.ModuleType("ideogram4_backend")

    class Ideogram4CogRunner:
        last_instance = None

        def __init__(self):
            self.build_s3_calls = []
            self.run_calls = []
            Ideogram4CogRunner.last_instance = self

        def _new_job_id(self):
            return "job-test"

        def build_s3_publishing_config(self, **kwargs):
            self.build_s3_calls.append(kwargs)
            return [{"base_path": kwargs.get("base_path") or "cog/ideogram4"}]

        def run(self, **kwargs):
            self.run_calls.append(kwargs)
            output_dir = Path(tempfile.mkdtemp())
            (output_dir / "adapter.safetensors").write_text("adapter", encoding="utf-8")
            return {"output_dir": str(output_dir), "nsfw_scan_summary": None, "nsfw_scan_report_path": None}

        def package_output(self, output_dir, *, target):
            Path(target).write_text("archive", encoding="utf-8")
            return target

        def read_debug_log(self):
            return ""

    backend.Ideogram4CogRunner = Ideogram4CogRunner
    sys.modules["ideogram4_backend"] = backend

    sys.modules.pop("predict", None)
    os.environ["SIMPLETUNER_COG_SKIP_CUDA_PREFLIGHT"] = "1"
    return importlib.import_module("predict"), Secret, Ideogram4CogRunner


class PredictorInputTests(unittest.TestCase):
    def test_predict_accepts_plain_string_secrets(self) -> None:
        predict, _, runner_cls = load_predict_module()
        predictor = predict.Predictor()
        predictor.setup()

        result = predictor.predict(
            train_data=None,
            hf_dataset="user/dataset",
            s3_bucket="bucket",
            s3_access_key="access",
            s3_secret_key="secret",
            hf_token="hf-token",
            return_logs=False,
        )

        self.assertTrue(Path(result).exists())
        runner = runner_cls.last_instance
        self.assertEqual(runner.build_s3_calls[0]["access_key"], "access")
        self.assertEqual(runner.build_s3_calls[0]["secret_key"], "secret")
        self.assertEqual(runner.run_calls[0]["hf_token"], "hf-token")
        self.assertEqual(runner.run_calls[0]["caption_strategy"], "textfile")
        self.assertIs(runner.run_calls[0]["enable_regional_compile"], True)

    def test_predict_accepts_secret_objects(self) -> None:
        predict, Secret, runner_cls = load_predict_module()
        predictor = predict.Predictor()
        predictor.setup()

        predictor.predict(
            train_data=None,
            hf_dataset="user/dataset",
            s3_bucket="bucket",
            s3_access_key=Secret("access-object"),
            s3_secret_key=Secret("secret-object"),
            hf_token=Secret("hf-token-object"),
            return_logs=False,
        )

        runner = runner_cls.last_instance
        self.assertEqual(runner.build_s3_calls[0]["access_key"], "access-object")
        self.assertEqual(runner.build_s3_calls[0]["secret_key"], "secret-object")
        self.assertEqual(runner.run_calls[0]["hf_token"], "hf-token-object")

    def test_predict_forwards_caption_strategy(self) -> None:
        predict, _, runner_cls = load_predict_module()
        predictor = predict.Predictor()
        predictor.setup()

        predictor.predict(
            train_data=None,
            hf_dataset="user/dataset",
            s3_bucket="bucket",
            caption_strategy="filename",
            return_logs=False,
        )

        runner = runner_cls.last_instance
        self.assertEqual(runner.run_calls[0]["caption_strategy"], "filename")

    def test_predict_can_disable_regional_compile(self) -> None:
        predict, _, runner_cls = load_predict_module()
        predictor = predict.Predictor()
        predictor.setup()

        predictor.predict(
            train_data=None,
            hf_dataset="user/dataset",
            s3_bucket="bucket",
            enable_regional_compile=False,
            return_logs=False,
        )

        runner = runner_cls.last_instance
        self.assertIs(runner.run_calls[0]["enable_regional_compile"], False)


if __name__ == "__main__":
    unittest.main()
