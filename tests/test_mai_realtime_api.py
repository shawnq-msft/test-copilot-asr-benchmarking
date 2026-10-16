import asyncio
import base64
import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from benchmark import asr_mai_realtime_api as api
from benchmark import config, runner
from benchmark.config import Sample
from scripts.compare_mai_transports import summarize, TRANSPORTS
from scripts import validate_benchmark


class FakeSocket:
    def __init__(self, fail=False):
        self.queue = asyncio.Queue()
        self.queue.put_nowait({"type": "session.created", "session": {"id": "session"}})
        self.response = SimpleNamespace(headers={"apim-request-id": "request"})
        self.messages = []
        self.fail = fail
        self.sends = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def recv(self):
        return json.dumps(await self.queue.get())

    async def send(self, raw):
        message = json.loads(raw)
        self.messages.append(message)
        kind = message["type"]
        if kind == "session.update":
            self.queue.put_nowait({"type": "session.updated", "session": message["session"]})
        elif kind == "input_audio_buffer.append":
            self.sends += len(base64.b64decode(message["audio"]))
            if self.fail:
                self.queue.put_nowait({"type": "error", "error": {"message": "test service failure"}})
            else:
                self.queue.put_nowait({"type": api.PREFIX + "intermediate", "intermediate": "Hello"})
        elif kind == "input_audio_buffer.commit":
            self.queue.put_nowait({"type": "input_audio_buffer.committed"})
            self.queue.put_nowait({"type": api.PREFIX + "delta", "delta": "Hello world", "item_id": "item"})
            self.queue.put_nowait({"type": api.PREFIX + "completed", "transcript": "Hello world",
                                  "item_id": "item"})


class ApiTests(unittest.TestCase):
    def test_preflight_chunk_validation_and_provenance(self):
        for chunk, status in ((0, 1), (20, 0)):
            output = io.StringIO()
            with patch("sys.argv", ["validate", "preflight", "--services", "whisper_v3",
                                    "--languages", "en-GB", "--chunk-ms", str(chunk)]), \
                    patch.object(validate_benchmark, "collect_samples",
                                 return_value=[Sample("s", "test", "en-GB", b"\0" * 6400, "hello", .2)]), \
                    redirect_stdout(output):
                self.assertEqual(validate_benchmark.main(), status)
            result = json.loads(output.getvalue())
            if status:
                self.assertIn("--chunk-ms must be positive", result["errors"])
            else:
                self.assertEqual(result["chunk_ms"], 20)

    def test_url_and_opt_in(self):
        settings = api.session_update("deployment")["session"]["audio"]["input"]
        self.assertEqual(settings["transcription"], {"model": "deployment"})
        self.assertEqual(api.realtime_url("https://resource.services.ai.azure.com"),
                         "wss://resource.services.ai.azure.com/mai/v1/realtime?intent=transcription")
        self.assertNotIn(api.SERVICE, config.DEFAULT_SERVICES)
        self.assertIn(api.SERVICE, config.SERVICES)
        self.assertIn(api.SERVICE, config.STREAMING_SERVICES)
        self.assertIn("realtime_mai_2", config.DEFAULT_SERVICES)
        self.assertIs(runner.ADAPTERS[api.SERVICE], api.transcribe_realtime_mai_2_api)
        for url in ("", "http://resource", "https://key@resource", "https://resource/path",
                    "https://resource?api-key=secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                api.realtime_url(url)

    def test_delta_and_intermediate_protocol(self):
        events = api.TranscriptEvents()
        for time, suffix, text in ((1, "delta", "Hello"), (2, "intermediate", " wor"),
                                   (3, "intermediate", " there"), (4, "delta", " there!")):
            events.accept({"type": api.PREFIX + suffix, suffix: text}, time)
        self.assertEqual(events.delta, "Hello there!")
        self.assertEqual(events.intermediate, "")
        self.assertEqual(events.first_text, 1)
        self.assertEqual(events.first_partial, 2)
        events.accept({"type": api.PREFIX + "completed", "transcript": "Hello there!"}, 5)
        self.assertEqual(events.transcript, "Hello there!")
        self.assertEqual(events.last_final, 5)

    def test_server_errors_and_missing_transcripts(self):
        for event in ({"type": "error", "error": {"message": "bad model"}},
                      {"type": api.PREFIX + "failed", "error": {"message": "failure"}},
                      {"type": api.PREFIX + "completed"}):
            with self.assertRaises(ValueError):
                api.TranscriptEvents().accept(event, 1)

    def test_append_commit_completion_and_evidence(self):
        async def exercise(fail):
            socket = FakeSocket(fail)
            sample = Sample("s", "en-US_test", "en-US", b"\0" * 640, "Hello world", .02, .001, .015)
            with patch.object(api, "connect", return_value=socket), \
                    patch.object(api, "MAI_REALTIME_ENDPOINT", "https://resource.services.ai.azure.com"), \
                    patch.object(api, "MAI_REALTIME_DEPLOYMENT", "model"):
                result = await api._transcribe(sample, True, 20)
            self.assertEqual(socket.sends, 640)
            self.assertEqual(result.session_id, "session")
            self.assertEqual(result.request_id, "apim-request-id=request")
            if fail:
                self.assertIn("test service failure", result.error)
                self.assertIsNone(result.upl_ms)
            else:
                self.assertIsNone(result.error)
                self.assertEqual(result.hypothesis, "Hello world")
                self.assertTrue(result.streaming_evidence["completed"])
                self.assertEqual(result.streaming_evidence["audio_bytes_sent"], 640)
                self.assertEqual(result.upl_anchor, "realtime")
                self.assertIsNotNone(result.first_latency_ms)
                self.assertGreaterEqual(result.streaming_evidence["end_of_audio_s"], .02)
                self.assertEqual(socket.messages[-1]["type"], "input_audio_buffer.commit")
                self.assertEqual(result.streaming_evidence["negotiated_input"]["turn_detection"], None)
        asyncio.run(exercise(False))
        asyncio.run(exercise(True))

    def test_summary_uses_matched_pairs_and_separates_chunks(self):
        rows = []
        for chunk in (20, 100):
            for repetition in (0, 1):
                for service, latency in zip(TRANSPORTS, (100, 80)):
                    rows.append({
                        "chunk_ms": chunk, "corpus": "xiaomi", "service": service, "repetition": repetition,
                        "dataset": "xiaomi", "sample_id": "s", "error": "failed" if repetition == 1 and service == TRANSPORTS[1] else "",
                        "setup_ms": 1, "first_text_from_audio_ms": latency, "first_text_from_speech_ms": latency,
                        "call_to_first_text_ms": latency + 1, "lbl_ms": latency, "upl_ms": latency, "wer": .1,
                    })
        summary = summarize(rows)
        self.assertEqual(summary["attempts"], 8)
        self.assertEqual(len(summary["failures"]), 2)
        for paired in summary["paired"]:
            self.assertEqual(paired["successful_pairs"], 1)
            metric = paired["metrics"]["first_text_from_audio_ms"]
            self.assertEqual(metric["api_minus_sdk"]["median"], -20)
            self.assertEqual(metric["api_lower_count"], 1)


if __name__ == "__main__":
    unittest.main()
