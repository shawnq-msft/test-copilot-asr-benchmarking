import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from benchmark import asr_whisper, config
from benchmark.config import Sample
from benchmark.eval_options import EvaluationOptions, validate_options
from scripts.validate_benchmark import check_resume, make_metadata, metadata_path


class AzureWhisperTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(asr_whisper, "_wait_for_request_slot"))
        self.sample = Sample("1", "recordings", "zh-CN", b"\0" * 6400, "", .2,
                             reference_available=False, evaluation_options=EvaluationOptions())
        for name, value in {
            "WHISPER_AZURE_ENDPOINT": "https://example.openai.azure.com",
            "WHISPER_AZURE_DEPLOYMENT": "Whisper",
            "WHISPER_AZURE_KEY": "test-key",
        }.items():
            self.enterContext(patch.object(config, name, value))

    def response(self, *, status=200, payload=None):
        response = requests.Response()
        response.status_code = status
        response.headers["apim-request-id"] = "request-1"
        response._content = json.dumps({"text": "你好"} if payload is None else payload).encode()
        return response

    def test_real_multipart_length_pacing_and_language(self):
        now = [0.0]
        body = asr_whisper._WhisperUpload(self.sample, True)
        prepared = requests.Request("POST", asr_whisper.azure_whisper_url(), data=body).prepare()
        self.assertEqual(int(prepared.headers["Content-Length"]), len(body))
        self.assertNotIn("Transfer-Encoding", prepared.headers)
        with patch.object(asr_whisper.time, "perf_counter", side_effect=lambda: now[0]), \
                patch.object(asr_whisper.time, "sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
            data = b"".join(body)
        self.assertEqual(len(data), len(body))
        self.assertIn(b'name="language"\r\n\r\nzh\r\n', data)
        self.assertIn(b"RIFF", data)
        self.assertAlmostEqual(body.end_of_audio - body.audio_start, .2)

    def test_success_and_unavailable_timestamps(self):
        def post(url, **kwargs):
            self.assertIn("/deployments/Whisper/audio/transcriptions?", url)
            b"".join(kwargs["data"])
            return self.response()

        with patch.object(asr_whisper.requests, "post", side_effect=post):
            result = asr_whisper.transcribe_azure_whisper(self.sample, pace=False)
        self.assertEqual(result.service, "azure_whisper")
        self.assertEqual(result.hypothesis, "你好")
        self.assertIsNone(result.error)
        self.assertIsNone(result.upl_ms)
        self.assertIsNone(result.segments[0]["language"])
        self.assertEqual(result.request_id, "apim-request-id=request-1")

    def test_failures_are_explicit(self):
        cases = [(429, {}, "status=429"), (200, {}, "missing string text"),
                 (200, {"text": ""}, "no_final"), (200, [], "missing string text")]
        for status, payload, message in cases:
            with self.subTest(status=status, payload=payload), \
                    patch.object(asr_whisper.requests, "post", return_value=self.response(status=status, payload=payload)):
                result = asr_whisper.transcribe_azure_whisper(self.sample, pace=False)
                self.assertIn(message, result.error)
                self.assertIn("request-1", result.request_id)
        with patch.object(asr_whisper.requests, "post", side_effect=requests.Timeout("timeout")):
            self.assertIn("http_error", asr_whisper.transcribe_azure_whisper(self.sample).error)
        with patch.object(asr_whisper.requests, "post", return_value=self.response()):
            self.assertIn("incomplete_upload", asr_whisper.transcribe_azure_whisper(self.sample).error)

    def test_preflight_identity_and_limits(self):
        self.assertNotIn("azure_whisper", config.DEFAULT_SERVICES)
        self.assertEqual(asr_whisper.transcribe_whisper_v3(self.sample).error, "not_implemented")
        config.assert_credentials(("azure_whisper",))
        with self.assertRaisesRegex(ValueError, "phrase"):
            validate_options(EvaluationOptions(), ["azure_whisper"], ["zh-CN"], phrases=["test"])
        with patch.object(config, "WHISPER_AZURE_ENDPOINT", "http://example.com"):
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                config.assert_credentials(("azure_whisper",))
        self.sample.pcm16_mono_16k = b"\0" * 25_000_000
        with patch.object(asr_whisper.requests, "post") as post:
            self.assertIn("25 MB", asr_whisper.transcribe_azure_whisper(self.sample).error)
            post.assert_not_called()

    def test_metadata_and_resume_bind_deployment(self):
        metadata = make_metadata([self.sample], ["azure_whisper"], pace=True, seed=42,
                                 limit=0, workers=1, phrases=[], phrase_weight=1,
                                 evaluation_options=EvaluationOptions())
        self.assertEqual(metadata["azure_whisper"]["model"], "whisper")
        self.assertEqual(metadata["azure_whisper"]["language_codes"], ["zh"])
        self.assertNotIn("test-key", json.dumps(metadata))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            path.touch()
            metadata_path(path).write_text(json.dumps(metadata))
            metadata["azure_whisper"]["deployment"] = "different"
            with self.assertRaisesRegex(ValueError, "deployment/model"):
                check_resume(path, metadata)

    def test_rate_limit_wait_is_outside_measured_upload(self):
        now = [0.0]
        with patch.object(asr_whisper, "_next_request", 0.0), \
                patch.object(config, "WHISPER_AZURE_REQUESTS_PER_MINUTE", 3), \
                patch.object(asr_whisper.time, "perf_counter", side_effect=lambda: now[0]), \
                patch.object(asr_whisper.time, "sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
            # Call the implementation while the test's general request stub is disabled.
            with patch.object(asr_whisper, "_wait_for_request_slot", self._wait_impl):
                asr_whisper._wait_for_request_slot()
                asr_whisper._wait_for_request_slot()
                asr_whisper._wait_for_request_slot()
            self.assertAlmostEqual(now[0], 40.2)

    _wait_impl = staticmethod(asr_whisper._wait_for_request_slot)


if __name__ == "__main__":
    unittest.main()
