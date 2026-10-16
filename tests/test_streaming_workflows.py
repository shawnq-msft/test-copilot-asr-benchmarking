from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmark import asr_realtime, config, datasets_loader, runner
from benchmark.audio_utils import write_pcm16_wav
from benchmark.config import AsrResult, Sample
from benchmark.report import build_report
from scripts import error_analysis, generate_testset, loose_match
from scripts.validate_benchmark import (check_resume, check_selection, collect_samples,
                                        make_metadata, metadata_path, validate_artifacts)


class Signal:
    def connect(self, callback):
        self.callback = callback

    def emit(self, event):
        self.callback(event)


class FakeSpeech:
    class SpeechConfig:
        model = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def request_word_level_timestamps(self):
            self.timestamps = True

        def set_property(self, *_args):
            pass

    ResultReason = SimpleNamespace(RecognizedSpeech="recognized")
    CancellationReason = SimpleNamespace(EndOfStream="eof")
    OutputFormat = SimpleNamespace(Detailed="detailed")
    PropertyId = SimpleNamespace(SpeechServiceResponse_RequestId="request",
                                 SpeechServiceResponse_PostProcessingOption="post",
                                 Speech_SegmentationSilenceTimeoutMs="silence")

    def __init__(self, scenario="success"):
        self.now = 10.0
        self.scenario = scenario
        self.writes = 0
        self.closed = False
        self.stopped = False
        self.audio = SimpleNamespace(AudioStreamFormat=lambda **kwargs: kwargs,
                                     PushAudioInputStream=lambda _fmt: self,
                                     AudioConfig=lambda **kwargs: kwargs)

    def SpeechRecognizer(self, *, speech_config, audio_config):
        self.config = speech_config
        self.audio_config = audio_config
        for name in ("recognizing", "recognized", "canceled", "session_started", "session_stopped"):
            setattr(self, name, Signal())
        return self

    def start_continuous_recognition_async(self):
        if self.scenario == "start_failure":
            raise RuntimeError("start rejected")
        self.session_started.emit(SimpleNamespace(session_id="session-1"))
        return SimpleNamespace(get=lambda: None)

    def stop_continuous_recognition_async(self):
        self.stopped = True
        if self.scenario == "stop_failure":
            raise RuntimeError("stop rejected")
        return SimpleNamespace(get=lambda: None)

    def event(self, text):
        return SimpleNamespace(session_id="session-1", result=SimpleNamespace(
            text=text, reason="recognized", result_id="result-1",
            properties=SimpleNamespace(get_property=lambda _prop: "request-1"),
            json=json.dumps({"NBest": [{"Words": [{"Word": text, "Offset": 200000,
                                                   "Duration": 1300000}]}]})))

    def write(self, _chunk):
        self.writes += 1
        self.now += 0.1
        self.recognizing.emit(self.event("partial"))
        if self.scenario == "write_failure":
            raise RuntimeError("write rejected")

    def close(self):
        self.closed = True
        if self.scenario == "start_failure":
            return
        if self.scenario == "timeout":
            return
        if self.scenario == "cancel":
            self.canceled.emit(SimpleNamespace(session_id="session-1", result=None,
                                               reason="Error", error_details="AuthenticationFailure"))
            return
        if self.scenario != "partial_only":
            self.now += 0.3
            self.recognized.emit(self.event("hello"))
            self.now += 0.1
            self.recognized.emit(self.event("world"))
        self.session_stopped.emit(SimpleNamespace(session_id="session-1"))


class StreamingTests(unittest.TestCase):
    def sample(self):
        return Sample("sample.wav", "en-GB_commands", "en-GB", b"\0" * 6400,
                      "hello world", 0.2, 0.02, 0.15)

    def run_fake(self, scenario="success", *, model=True, anchored=True):
        fake = FakeSpeech(scenario)
        sample = self.sample()
        if not anchored:
            sample.first_word_start_s = sample.last_word_end_s = None
            fake.event = lambda text: SimpleNamespace(session_id="session-1", result=SimpleNamespace(
                text=text, reason="recognized", result_id="result-1",
                properties=SimpleNamespace(get_property=lambda _prop: ""),
                json="{}"))
        with patch.object(asr_realtime, "speechsdk", fake), \
                patch.object(asr_realtime.time, "perf_counter", side_effect=lambda: fake.now), \
                patch.object(asr_realtime.time, "sleep") as sleep:
            if scenario == "timeout":
                with patch.object(asr_realtime.threading, "Event") as event:
                    event.return_value.is_set.return_value = False
                    event.return_value.wait.return_value = False
                    result = self.invoke(sample, model)
            else:
                result = self.invoke(sample, model)
            self.assertEqual(sleep.call_count, 0)
        return result, fake

    def invoke(self, sample, model):
        return asr_realtime._run_realtime(
            sample, service_name="realtime_mai_2" if model else "realtime",
            key="fake-key", region="region", endpoint="", pace=True, post_refinement=False,
            model=config.MAI_STREAMING_MODEL if model else None)

    def test_model_and_exact_latency(self):
        result, fake = self.run_fake()
        self.assertEqual(fake.config.model, "mai-transcribe-2-streaming")
        self.assertFalse(hasattr(fake.config, "timestamps"))
        self.assertFalse(hasattr(fake.config, "speech_recognition_language"))
        self.assertEqual(result.hypothesis, "hello world")
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.first_latency_ms, 80)
        self.assertAlmostEqual(result.lbl_ms, 400)
        self.assertAlmostEqual(result.upl_ms, 450)
        self.assertEqual(result.upl_anchor, "realtime")
        self.assertEqual(result.request_id, "RequestId=request-1")
        self.assertEqual(result.session_id, "session-1")
        self.assertEqual(result.streaming_evidence["final_count"], 2)
        self.assertTrue(fake.closed and fake.stopped)

    def test_baseline_configuration_preserved(self):
        result, fake = self.run_fake(model=False)
        self.assertTrue(fake.config.timestamps)
        self.assertEqual(fake.config.speech_recognition_language, "en-GB")
        self.assertIsNone(result.error)

    def test_matched_chunk_packetization_and_pacing(self):
        for chunk_ms, count in ((20, 10), (100, 2)):
            with self.subTest(chunk_ms=chunk_ms):
                fake = FakeSpeech()
                packets = []

                def write(chunk):
                    packets.append(chunk)
                    fake.now += .001
                    fake.recognizing.emit(fake.event("partial"))

                def sleep(seconds):
                    fake.now += seconds

                fake.write = write
                with patch.object(asr_realtime, "speechsdk", fake), \
                        patch.object(asr_realtime.time, "perf_counter", side_effect=lambda: fake.now), \
                        patch.object(asr_realtime.time, "sleep", side_effect=sleep):
                    result = asr_realtime._run_realtime(
                        self.sample(), service_name="realtime_mai_2", key="fake-key", region="region",
                        endpoint="", pace=True, post_refinement=False,
                        model=config.MAI_STREAMING_MODEL, chunk_ms=chunk_ms)
                self.assertIsNone(result.error)
                self.assertEqual(len(packets), count)
                self.assertEqual(sum(map(len, packets)), 6400)
                self.assertEqual(result.streaming_evidence["chunk_ms"], chunk_ms)
                self.assertAlmostEqual(result.streaming_evidence["end_of_audio_s"], .2)

    def test_runner_preserves_selected_streaming_order_after_baseline(self):
        order = []

        def call(adapter, sample, pace, retries, phrases, weight, chunk):
            service = next(name for name, implementation in runner.ADAPTERS.items()
                           if implementation == adapter)
            order.append(service)
            self.assertEqual(chunk, 20)
            return AsrResult(service, "hello world", None, None, None)

        with patch.object(runner, "_call_with_retries", side_effect=call):
            runner._run_sample(self.sample(), ("realtime_mai_2_api", "realtime_mai_2", "realtime"),
                               True, 0, chunk_ms=20)
        self.assertEqual(order, ["realtime", "realtime_mai_2_api", "realtime_mai_2"])

    def test_mai_only_without_words_has_no_invented_anchor(self):
        result, _ = self.run_fake(anchored=False)
        self.assertIsNone(result.first_latency_ms)
        self.assertIsNone(result.upl_ms)
        self.assertIsNone(result.upl_anchor)
        self.assertIsNotNone(result.lbl_ms)
        self.assertEqual(result.request_id, "ResultId=result-1")

    def test_failures_are_explicit(self):
        for scenario, message in (("cancel", "AuthenticationFailure"), ("timeout", "timeout"),
                                  ("partial_only", "no_final"), ("write_failure", "stream_failed"),
                                  ("start_failure", "start_failed"), ("stop_failure", "stop_failed")):
            with self.subTest(scenario=scenario):
                result, fake = self.run_fake(scenario)
                self.assertIn(message, result.error)
                self.assertIsNone(result.first_latency_ms)
                self.assertTrue(fake.closed)

    def test_unverified_phrase_option_rejected(self):
        result = asr_realtime.transcribe_realtime_mai_2(self.sample(), phrases=("hint",))
        self.assertIn("Phrase-list", result.error)

    def test_model_registration(self):
        self.assertIn("realtime_mai_2", config.SERVICES)
        self.assertIn("realtime_mai_2", config.DEFAULT_SERVICES)
        self.assertIn("realtime_mai_2", config.STREAMING_SERVICES)
        self.assertIs(runner.ADAPTERS["realtime_mai_2"], asr_realtime.transcribe_realtime_mai_2)

    def test_retired_model_is_removed_without_substitution(self):
        import inspect
        from benchmark import asr_llm_speech
        from benchmark.eval_options import CAPABILITIES

        self.assertNotIn("fast_mai", config.SERVICES)
        self.assertNotIn("fast_mai", config.DEFAULT_SERVICES)
        self.assertNotIn("fast_mai", runner.ADAPTERS)
        self.assertNotIn("fast_mai", CAPABILITIES)
        self.assertFalse(hasattr(asr_llm_speech, "transcribe_fast_mai"))
        self.assertEqual(inspect.signature(runner.run_benchmark).parameters["services"].default,
                         config.DEFAULT_SERVICES)
        with self.assertRaisesRegex(ValueError, "known service"):
            check_selection(["fast_mai"], ["en-GB"], 1, 1, [])
        with self.assertRaisesRegex(ValueError, "Unknown service"):
            config.assert_credentials(("fast_mai",))
        for service, model in (("fast_mai_1_5", "MAI-Transcribe-1.5"), ("fast_mai_2", "MAI-Transcribe-2")):
            self.assertIn(service, config.SERVICES)
            self.assertIn(service, config.DEFAULT_SERVICES)
            self.assertIn(service, runner.ADAPTERS)
            with patch.object(asr_llm_speech, "_streaming_upload") as upload:
                runner.ADAPTERS[service](self.sample())
            self.assertEqual(upload.call_args.args[1]["enhancedMode"]["model"], model)

    def test_removed_service_cli_is_rejected_before_running(self):
        import run_full

        with patch("sys.argv", ["run_full.py", "--services", "fast_mai"]), \
                patch.object(run_full, "run_benchmark") as run, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            run_full.main()
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()

    def test_region_warnings_distinguish_rest_from_streaming(self):
        services = ["fast_llm", "fast_mai_1_5", "fast_mai_2", "realtime_mai_2"]
        with patch.multiple(config, AZURE_SPEECH_REGION="eastus2", LLM_AZURE_REGION="eastus2"):
            warnings = check_selection(services, ["en-US"], 0, 1, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("REST enhanced mode", warnings[0])
        self.assertIn("HTTP 400", warnings[0])
        with patch.multiple(config, AZURE_SPEECH_REGION="southeastasia",
                            LLM_AZURE_REGION="southeastasia"):
            self.assertEqual(check_selection(services, ["en-US"], 0, 1, []), [])
        with patch.multiple(config, AZURE_SPEECH_REGION="eastus", LLM_AZURE_REGION="eastus"):
            warnings = check_selection(services, ["en-US"], 0, 1, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("MAI streaming", warnings[0])

    def test_old_sdk_cannot_silently_ignore_model(self):
        fake = FakeSpeech()
        fake.SpeechConfig = lambda **_kwargs: None
        with patch.object(asr_realtime, "speechsdk", fake):
            result = self.invoke(self.sample(), True)
        self.assertIn(">=1.52.0", result.error)

    def test_invalid_word_json(self):
        for text in ("not json", "[]", '{"NBest":[{"Words":[{"Offset":-1,"Duration":1}]}]}',
                     '{"NBest":"invalid"}', '{"NBest":[{"Words":["invalid"]}]}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                asr_realtime._extract_words(text)

    def test_pacing_flag_controls_sleep(self):
        for pace in (True, False):
            fake = FakeSpeech()
            original_write = fake.write

            def quick_write(chunk):
                original_write(chunk)
                fake.now -= 0.09

            def sleep(seconds):
                fake.now += seconds

            fake.write = quick_write
            with patch.object(asr_realtime, "speechsdk", fake), \
                    patch.object(asr_realtime.time, "perf_counter", side_effect=lambda: fake.now), \
                    patch.object(asr_realtime.time, "sleep", side_effect=sleep) as sleeping:
                result = asr_realtime._run_realtime(
                    self.sample(), service_name="realtime_mai_2", key="fake", region="region",
                    endpoint="", pace=pace, post_refinement=False, model=config.MAI_STREAMING_MODEL)
                self.assertEqual(sleeping.call_count, 2 if pace else 0)
                self.assertEqual(result.streaming_evidence["pace"], pace)
                self.assertEqual(result.streaming_evidence["audio_bytes_sent"], 6400)

    def test_premature_session_end_is_failure_even_if_finals_arrive(self):
        fake = FakeSpeech()
        original_write = fake.write

        def stop_early(chunk):
            original_write(chunk)
            fake.session_stopped.emit(SimpleNamespace(session_id="session-1"))

        fake.write = stop_early
        with patch.object(asr_realtime, "speechsdk", fake), \
                patch.object(asr_realtime.time, "perf_counter", side_effect=lambda: fake.now):
            result = self.invoke(self.sample(), True)
        self.assertIn("premature_end", result.error)
        self.assertIsNone(result.upl_ms)
        self.assertEqual(result.streaming_evidence["audio_bytes_sent"], 3200)

    def test_streaming_services_share_primary_credentials(self):
        with patch.multiple(config, AZURE_SPEECH_KEY="shared",
                            AZURE_SPEECH_REGION="centralus", AZURE_SPEECH_ENDPOINT=""):
            config.assert_credentials(("realtime", "realtime_mai_2"))
        with patch.object(config, "AZURE_SPEECH_KEY", None):
            for service in ("realtime", "realtime_mai_2"):
                with self.subTest(service=service), self.assertRaisesRegex(ValueError, service + ":"):
                    config.assert_credentials((service,))

    def test_mai_adapter_uses_baseline_resource(self):
        with patch.multiple(asr_realtime, AZURE_SPEECH_KEY="shared",
                            AZURE_SPEECH_REGION="centralus", AZURE_SPEECH_ENDPOINT=""), \
                patch.object(asr_realtime, "_run_realtime") as run:
            asr_realtime.transcribe_realtime_mai_2(self.sample())
        self.assertEqual(run.call_args.kwargs["key"], "shared")
        self.assertEqual(run.call_args.kwargs["region"], "centralus")
        self.assertEqual(run.call_args.kwargs["endpoint"], "")
        self.assertEqual(run.call_args.kwargs["model"], config.MAI_STREAMING_MODEL)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv = self.root / "test.csv"
        self.sample = Sample("a.wav", "en-GB_commands", "en-GB", b"\0" * 16000,
                             "air flow", 0.5)
        self.services = ["realtime_mai_2"]
        self.metadata = make_metadata([self.sample], self.services, pace=True,
                                      seed=42, limit=1, workers=1, phrases=[], phrase_weight=1.0)
        metadata_path(self.csv).write_text(json.dumps(self.metadata), encoding="utf-8")
        self.result = AsrResult("realtime_mai_2", "airflow", None, 10,
                                request_id="req", session_id="session",
                                streaming_evidence={"completed": True, "final_count": 1,
                                                    "partial_count": 1, "pace": True})
        self.stdout = redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def run_fixture(self):
        with patch.dict(runner.ADAPTERS, {"realtime_mai_2": Mock(return_value=self.result)}):
            runner.run_benchmark([self.sample], self.csv, services=tuple(self.services), workers=1, retries=0)

    def test_removed_service_historical_results_remain_readable(self):
        metadata = make_metadata([self.sample], ["fast_mai"], pace=True, seed=42,
                                 limit=1, workers=1, phrases=[], phrase_weight=1.0)
        metadata_path(self.csv).write_text(json.dumps(metadata), encoding="utf-8")
        result = AsrResult("fast_mai", "air flow", None, 10, request_id="historical-request")
        with self.csv.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=runner.CSV_FIELDS)
            writer.writeheader()
            writer.writerow(runner._to_row(self.sample, result))
        with patch("benchmark.report.tcp_ping_ms", return_value=None), \
                patch("benchmark.report.public_ip_and_location", return_value={}), \
                patch("scripts.error_analysis._load_sample_index", return_value={}):
            report = self.csv.with_name("test_report.md")
            analysis = self.csv.with_name("test_error_analysis.md")
            build_report(self.csv, report)
            error_analysis.main(self.csv, analysis)
        self.assertIn("`fast_mai` — historical/unregistered service", report.read_text(encoding="utf-8"))
        self.assertIn("fast_mai", analysis.read_text(encoding="utf-8"))
        self.assertTrue(validate_artifacts(self.csv)["ok"])

    def test_exact_artifacts_normalization_and_reports(self):
        self.run_fixture()
        before = self.csv.read_bytes()
        self.assertTrue(validate_artifacts(self.csv, require_reports=False)["ok"])
        loose_match.main(self.csv)
        loose_match.main(self.csv)
        self.assertEqual(self.csv.with_suffix(".csv.strict").read_bytes(), before)
        with patch("benchmark.report.tcp_ping_ms", return_value=None), \
                patch("benchmark.report.public_ip_and_location", return_value={}), \
                patch("scripts.error_analysis._load_sample_index", return_value={}):
            report = self.csv.with_name("test_report.md")
            analysis = self.csv.with_name("test_error_analysis.md")
            build_report(self.csv, report)
            error_analysis.main(self.csv, analysis)
        self.assertIn("MAI-Transcribe-2-Streaming", report.read_text(encoding="utf-8"))
        self.assertIn("No baseline realtime", report.read_text(encoding="utf-8"))
        self.assertIn("First Latency", analysis.read_text(encoding="utf-8"))
        self.assertIn("Best / median / worst", analysis.read_text(encoding="utf-8"))
        self.assertTrue(validate_artifacts(self.csv)["ok"])
        with self.assertRaisesRegex(ValueError, "normalized"):
            check_resume(self.csv, self.metadata)

    def test_resume_does_not_repeat_completed_calls_or_sidecars(self):
        self.run_fixture()
        sidecar = self.csv.with_name("test_words.jsonl")
        original = sidecar.read_bytes()
        check_resume(self.csv, self.metadata)
        with patch.dict(runner.ADAPTERS, {"realtime_mai_2": Mock(side_effect=AssertionError("repeated"))}):
            runner.run_benchmark([self.sample], self.csv, services=tuple(self.services), workers=1)
        self.assertEqual(sidecar.read_bytes(), original)
        self.assertTrue(validate_artifacts(self.csv, require_reports=False)["ok"])

    def test_invalid_coverage_and_service_failures(self):
        self.run_fixture()
        with self.csv.open("a", encoding="utf-8", newline="") as dest:
            csv.DictWriter(dest, fieldnames=runner.CSV_FIELDS).writerow(runner._to_row(self.sample, self.result))
        self.assertFalse(validate_artifacts(self.csv, require_reports=False)["ok"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            check_resume(self.csv, self.metadata)
        self.csv.unlink()
        self.csv.with_name("test_words.jsonl").unlink()
        self.result.error = "service unavailable"
        self.run_fixture()
        outcome = validate_artifacts(self.csv, require_reports=False)
        self.assertFalse(outcome["ok"])
        self.assertEqual(len(outcome["service_errors"]), 1)
        self.assertTrue(outcome["coverage_verified"])

    def test_missing_expected_rows_and_evidence(self):
        self.run_fixture()
        self.metadata["expected_keys"].append(["en-GB_commands", "missing.wav", "realtime_mai_2"])
        metadata_path(self.csv).write_text(json.dumps(self.metadata), encoding="utf-8")
        self.csv.with_name("test_words.jsonl").unlink()
        outcome = validate_artifacts(self.csv, require_reports=False)
        self.assertFalse(outcome["coverage_verified"])
        self.assertTrue(any("evidence" in error for error in outcome["errors"]))
        self.assertEqual(len(outcome["missing_keys"]), 1)

    def test_legacy_analysis_requires_explicit_validation_opt_in(self):
        self.run_fixture()
        metadata_path(self.csv).unlink()
        self.assertFalse(validate_artifacts(self.csv, require_reports=False)["ok"])
        result = validate_artifacts(self.csv, require_reports=False, allow_legacy=True)
        self.assertTrue(result["ok"])
        self.assertFalse(result["coverage_verified"])

    def test_selection_guards_and_unpaced_provenance(self):
        for services, locales, limit, workers in (([], ["en-GB"], 1, 1),
                                                  (["typo"], ["en-GB"], 1, 1),
                                                  (self.services, [], 1, 1),
                                                  (self.services, ["en-GB"], -1, 1),
                                                  (self.services, ["en-GB"], 1, 0)):
            with self.assertRaises(ValueError):
                check_selection(services, locales, limit, workers, [])
        self.run_fixture()
        self.metadata["pace"] = False
        metadata_path(self.csv).write_text(json.dumps(self.metadata), encoding="utf-8")
        outcome = validate_artifacts(self.csv, require_reports=False)
        self.assertTrue(any("Unpaced" in warning for warning in outcome["warnings"]))

    def test_loader_checks_exact_references_not_wav_counts(self):
        data = self.root / "en-GB" / "commands"
        data.mkdir(parents=True)
        (data / "trans.txt").write_text("wanted.wav\thello\n", encoding="utf-8")
        write_pcm16_wav(self.sample.pcm16_mono_16k, data / "unrelated.wav")
        with patch.object(datasets_loader, "DATA_ROOT", self.root):
            with self.assertRaises(FileNotFoundError):
                collect_samples(["en-GB"], 1, 42)
            write_pcm16_wav(self.sample.pcm16_mono_16k, data / "wanted.wav")
            self.assertEqual(len(collect_samples(["en-GB"], 1, 42)), 1)

    def test_generation_failure_is_not_success(self):
        data = self.root / "en-GB" / "commands"
        data.mkdir(parents=True)
        (data / "trans.txt").write_text("wanted.wav\thello\n", encoding="utf-8")
        with patch.object(generate_testset, "DATA_ROOT", self.root), \
                patch.object(generate_testset, "_synthesize", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "TTS failures"):
                generate_testset.process_dataset("en-GB", "commands", "voice")

    def test_cli_end_to_end_and_resume(self):
        import run_full

        data = self.root / "input" / "en-GB" / "commands"
        data.mkdir(parents=True)
        (data / "trans.txt").write_text("a.wav\tair flow\n", encoding="utf-8")
        write_pcm16_wav(self.sample.pcm16_mono_16k, data / "a.wav")
        output = self.root / "output"
        argv = ["run_full.py", "--languages", "en-GB", "--services", "realtime_mai_2",
                "--language-mode", "multilingual", "--lid-strategy", "native",
                "--max-per-dataset", "1", "--workers", "1",
                "--testdata-dir", str(data.parent.parent), "--output-dir", str(output)]
        adapter = Mock(return_value=self.result)
        with patch.object(run_full, "assert_credentials"), \
                patch("benchmark.report.tcp_ping_ms", return_value=None), \
                patch("benchmark.report.public_ip_and_location", return_value={}), \
                patch.dict(runner.ADAPTERS, {"realtime_mai_2": adapter}), \
                patch.object(datasets_loader, "DATA_ROOT", data.parent.parent):
            with patch("sys.argv", argv):
                run_full.main()
            csv_path, = output.glob("*.csv")
            self.assertTrue(validate_artifacts(csv_path)["ok"])
            with patch("sys.argv", argv + ["--resume", str(csv_path)]):
                run_full.main()
            self.assertEqual(adapter.call_count, 1)
            strict_bytes = csv_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "strict token ownership"):
                loose_match.main(csv_path)
            build_report(csv_path, csv_path.with_name(csv_path.stem + "_report.md"))
            error_analysis.main(csv_path, csv_path.with_name(csv_path.stem + "_error_analysis.md"))
            self.assertTrue(validate_artifacts(csv_path)["ok"])
            self.assertEqual(csv_path.read_bytes(), strict_bytes)

    def test_partial_resume_restores_baseline_without_calling_it(self):
        baseline = AsrResult("realtime", "air flow", 20, 10, request_id="r", session_id="s",
                             words=[{"text": "air", "start_s": 0.01, "end_s": 0.2},
                                    {"text": "flow", "start_s": 0.2, "end_s": 0.4}])
        with patch.object(runner, "transcribe_realtime", return_value=baseline):
            runner.run_benchmark([self.sample], self.csv, services=("realtime",), workers=1)
        self.sample.first_word_start_s = self.sample.last_word_end_s = None
        adapter = Mock(return_value=self.result)
        with patch.object(runner, "transcribe_realtime", side_effect=AssertionError("repeat")), \
                patch.dict(runner.ADAPTERS, {"realtime_mai_2": adapter}):
            runner.run_benchmark([self.sample], self.csv,
                                 services=("realtime", "realtime_mai_2"), workers=1)
        self.assertEqual(adapter.call_count, 1)
        self.assertEqual(self.sample.first_word_start_s, 0.01)
        self.assertEqual(self.sample.last_word_end_s, 0.4)

    def test_mixed_service_disagreement_and_first_latency_display(self):
        self.result.first_latency_ms = 123
        rest = AsrResult("fast_default", "air flow", None, 5, request_id="rest-id")
        with patch.dict(runner.ADAPTERS, {"realtime_mai_2": Mock(return_value=self.result),
                                        "fast_default": Mock(return_value=rest)}):
            runner.run_benchmark([self.sample], self.csv,
                                 services=("fast_default", "realtime_mai_2"), workers=1)
        with patch("benchmark.report.tcp_ping_ms", return_value=None), \
                patch("benchmark.report.public_ip_and_location", return_value={}), \
                patch("scripts.error_analysis._load_sample_index", return_value={}):
            report = self.root / "test_report.md"
            analysis = self.root / "test_error_analysis.md"
            build_report(self.csv, report)
            error_analysis.main(self.csv, analysis)
        self.assertIn("123 / 123", report.read_text(encoding="utf-8"))
        self.assertIn("123 / 123", analysis.read_text(encoding="utf-8"))
        self.assertIn("(fast_default=0.000, realtime_mai_2=1.000)", analysis.read_text(encoding="utf-8"))

    def test_legacy_untagged_words_remain_readable(self):
        self.result.service = "realtime"
        with self.csv.open("w", encoding="utf-8", newline="") as dest:
            writer = csv.DictWriter(dest, fieldnames=runner.CSV_FIELDS)
            writer.writeheader()
            writer.writerow(runner._to_row(self.sample, self.result))
        metadata_path(self.csv).unlink()
        self.csv.with_name("test_words.jsonl").write_text(json.dumps({
            "dataset": self.sample.dataset, "sample_id": self.sample.sample_id,
            "words": [], "boundary_fix": {"action": "skip"},
        }), encoding="utf-8")
        outcome = validate_artifacts(self.csv, require_reports=False, allow_legacy=True)
        self.assertTrue(outcome["ok"])
        self.assertFalse(outcome["coverage_verified"])

    def test_generation_and_preflight_select_identical_subset(self):
        data = self.root / "en-GB" / "commands"
        data.mkdir(parents=True)
        (data / "trans.txt").write_text("a.wav\tone\nb.wav\ttwo\nc.wav\tthree\n", encoding="utf-8")
        generated = []

        def synthesize(_text, _voice, output):
            generated.append(output.name)
            write_pcm16_wav(self.sample.pcm16_mono_16k, output)
            return True

        with patch.object(generate_testset, "DATA_ROOT", self.root), \
                patch.object(datasets_loader, "DATA_ROOT", self.root), \
                patch.object(generate_testset, "_synthesize", side_effect=synthesize):
            generate_testset.process_dataset("en-GB", "commands", "voice", limit=1, seed=42)
            samples = collect_samples(["en-GB"], 1, 42)
            self.assertEqual(len(generated), 1)
            self.assertEqual(samples[0].sample_id, f"commands/{generated[0]}")
            with self.assertRaises(FileNotFoundError):
                collect_samples(["en-GB"], 0, 42)

    def test_skill_entry_points_and_manual_justification(self):
        root = Path(__file__).resolve().parents[1]
        skill = root / ".github" / "skills" / "asr-benchmark" / "SKILL.md"
        self.assertIn("name: asr-benchmark", skill.read_text(encoding="utf-8"))
        for prompt in (root / ".github" / "prompts").glob("0[1-5]-*.prompt.md"):
            self.assertIn("../skills/asr-benchmark/SKILL.md", prompt.read_text(encoding="utf-8"))
        justify = root / ".github" / "skills" / "asr-justify" / "SKILL.md"
        self.assertIn("disable-model-invocation: true", justify.read_text(encoding="utf-8"))

    def test_cli_and_preflight_use_active_defaults(self):
        import run_full
        from scripts import validate_benchmark

        with patch("sys.argv", ["run_full.py", "--languages", "en-US",
                                "--output-dir", str(self.root)]), \
                patch.object(run_full, "assert_credentials"), \
                patch.object(run_full, "check_selection", wraps=check_selection) as selection, \
                patch.object(run_full, "collect_samples", side_effect=RuntimeError("stop before network")):
            with self.assertRaisesRegex(RuntimeError, "stop before network"):
                run_full.main()
            self.assertEqual(selection.call_args.args[0], list(config.DEFAULT_SERVICES))
        with patch("sys.argv", ["validate_benchmark.py", "preflight", "--languages", "en-US",
                                "--output-dir", str(self.root)]), \
                patch.object(validate_benchmark, "assert_credentials"), \
                patch.object(validate_benchmark, "collect_samples", return_value=[self.sample]), \
                patch.object(validate_benchmark, "check_selection", wraps=check_selection) as selection:
            self.assertEqual(validate_benchmark.main(), 1)
            self.assertEqual(selection.call_args.args[0], list(config.DEFAULT_SERVICES))


if __name__ == "__main__":
    unittest.main()
