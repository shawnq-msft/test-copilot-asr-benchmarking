import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_full
from benchmark import asr_fast, asr_realtime, config, datasets_loader, runner
from benchmark.annotation_metrics import language_scores, speaker_agreement
from benchmark.audio_utils import write_pcm16_wav
from benchmark.config import AsrResult, Sample
from benchmark.eval_options import EvaluationOptions, validate_options
from benchmark.evaluation import evaluation_path, variant_id, evaluation_record
from benchmark.report import build_report
from scripts import error_analysis, loose_match
from scripts.validate_benchmark import make_metadata, metadata_path, check_resume, validate_artifacts
from test_streaming_workflows import FakeSpeech


def segment(text, speaker=None, language=None, start=None, end=None):
    return {"text": text, "speaker": speaker, "language": language, "start_s": start, "end_s": end}


class MetricTests(unittest.TestCase):
    def test_exact_wder_permutation_and_one_of_four(self):
        ref = [segment("One two", "a"), segment("three four", "b")]
        self.assertEqual(speaker_agreement(ref, [segment("one two", "x"), segment("three four", "y")])["rate"], 0)
        result = speaker_agreement(ref, [segment("one two", "x"), segment("three", "y"), segment("four", "x")])
        self.assertEqual(result["rate"], .25)
        self.assertEqual(result["speaker_errors"], 1)
        self.assertEqual(result["aligned_words"], 4)
        self.assertEqual(result["alignment_coverage"], 1)

    def test_unknown_fragmentation_substitution_and_deletion(self):
        ref = [segment("one two three four", "a")]
        for hyp, expected in (
            ([segment("one two three four", "Unknown")], 1),
            ([segment("one two", "x"), segment("three four", "y")], .5),
            ([segment("one wrong three four", "x")], 0),
            ([segment("one two", "x")], 0),
            ([segment("one two three four extra", "x")], 0),
        ):
            with self.subTest(hyp=hyp):
                self.assertEqual(speaker_agreement(ref, hyp)["rate"], expected)
        self.assertIsNone(speaker_agreement(ref, [])["rate"])
        self.assertIsNone(speaker_agreement([], [])["rate"])
        self.assertEqual(speaker_agreement(ref, [segment("one two", "x")])["alignment_coverage"], .5)

    def test_normalization_and_repeated_word_alignment_is_deterministic(self):
        ref = [segment("Hello, hello!", "a"), segment("HELLO there.", "b")]
        hyp = [segment("hello hello", "q"), segment("hello there", "r")]
        self.assertEqual(speaker_agreement(ref, hyp)["rate"], 0)
        self.assertEqual(speaker_agreement(ref, hyp), speaker_agreement(ref, hyp))

    def test_monolingual_not_lid_and_language_not_locale(self):
        predicted = [segment("hello", language="en-US")]
        self.assertIsNone(language_scores("en-GB", [], predicted, None)["clip_correct"])
        scores = language_scores("en-GB", [], predicted, "at-start")
        self.assertTrue(scores["clip_correct"])
        self.assertFalse(scores["clip_locale_exact"])
        self.assertIsNone(language_scores("en-GB", [], [segment("hello")], "at-start")["clip_correct"])

    def test_segment_language_scoring_and_missing_coverage(self):
        ref = [segment("hi", language="en-GB", start=0, end=1),
               segment("hallo", language="de-DE", start=1, end=2)]
        pred = [segment("hi hallo", language="en-US", start=0, end=2)]
        score = language_scores(None, ref, pred, "continuous")
        self.assertEqual(score["segment_accuracy"], .5)
        self.assertEqual(score["segment_coverage"], 1)
        missing = language_scores(None, ref, pred[:0], "continuous")
        self.assertIsNone(missing["segment_accuracy"])
        partial = language_scores(None, ref, [segment("hi", language="en-US", start=0, end=1)], "native")
        self.assertEqual(partial["segment_coverage"], .5)
        self.assertEqual(partial["segment_accuracy"], .5)


class OptionsTests(unittest.TestCase):
    def test_default_and_explicit_strategy(self):
        validate_options(EvaluationOptions(), ["fast_default", "fast_llm", "fast_mai_2"], ["en-GB"])
        with self.assertRaisesRegex(ValueError, "explicit"):
            validate_options(EvaluationOptions(language_mode="multilingual"), ["realtime"], ["en-GB"])
        validate_options(EvaluationOptions(), ["realtime_mai_2"], ["en-GB"])
        validate_options(EvaluationOptions("multilingual", "native"), ["realtime_mai_2"], [])
        self.assertNotIn("realtime_diarization", config.DEFAULT_SERVICES)

    def test_invalid_combinations(self):
        invalid = [
            (EvaluationOptions(diarization=True), ["realtime"]),
            (EvaluationOptions(score_wder=True), ["fast_default"]),
            (EvaluationOptions("multilingual", "at-start", ("en-GB", "en-US")), ["realtime"]),
            (EvaluationOptions("multilingual", "native", ("en-GB",)), ["fast_default"]),
            (EvaluationOptions(max_speakers=4), ["fast_default"]),
            (EvaluationOptions(diarization=True, max_speakers=4), ["realtime_diarization"]),
            (EvaluationOptions("multilingual", "continuous", ("en-GB", "de-DE"), True), ["realtime_diarization"]),
        ]
        for options, services in invalid:
            with self.subTest(options=options, services=services), self.assertRaises(ValueError):
                validate_options(options, services, ["en-GB"])

    def test_rest_language_payload_and_no_truncation(self):
        sample = Sample("s", "test", "en-GB", b"\0" * 6400, "hello", .2, 0, .05)
        response = SimpleNamespace(status_code=200, headers={"apim-request-id": "id"},
                                   content=json.dumps({"combinedPhrases": [{"text": "hello"}],
                                                       "phrases": [{"text": "hello", "speaker": 0, "locale": "en-GB",
                                                                    "offsetMilliseconds": 0, "durationMilliseconds": 200}]}).encode())
        bodies = []

        def post(*args, **kwargs):
            bodies.append(b"".join(kwargs["data"]))
            return response

        for options in (EvaluationOptions(), EvaluationOptions("multilingual", "at-start", ("en-GB", "de-DE")),
                        EvaluationOptions("multilingual", "native", diarization=True, max_speakers=3)):
            sample.evaluation_options = options
            with patch.object(asr_fast.requests, "post", side_effect=post):
                result = asr_fast.transcribe_fast_default(sample, pace=False)
            self.assertIsNone(result.error)
            self.assertEqual(result.segments[0]["speaker"], 0)
            self.assertEqual(result.segments[0]["language"], "en-GB")
            if options.language_mode == "multilingual":
                self.assertIsNone(result.vad_truncated_s)
            if options.lid_strategy == "native":
                self.assertNotIn(b'"locales"', bodies[-1])
                self.assertIn(b'"maxSpeakers": 3', bodies[-1])
            else:
                self.assertIn(b'"locales": [', bodies[-1])

    def test_sdk_diarization_shared_lifecycle_and_speaker(self):
        fake = FakeSpeech()
        original = fake.SpeechRecognizer

        def transcriber(**kwargs):
            obj = original(**kwargs)
            obj.transcribing, obj.transcribed = obj.recognizing, obj.recognized
            obj.start_transcribing_async = obj.start_continuous_recognition_async
            obj.stop_transcribing_async = obj.stop_continuous_recognition_async
            return obj

        fake.transcription = SimpleNamespace(ConversationTranscriber=transcriber)
        event = fake.event

        def labeled_event(text):
            result = event(text)
            result.result.speaker_id = "Guest-1"
            return result

        fake.event = labeled_event
        sample = Sample("s", "test", "en-GB", b"\0" * 6400, "hello world", .2,
                        evaluation_options=EvaluationOptions(diarization=True))
        with patch.object(asr_realtime, "speechsdk", fake), \
                patch.object(asr_realtime.time, "perf_counter", side_effect=lambda: fake.now), \
                patch.object(asr_realtime.time, "sleep"):
            result = asr_realtime.transcribe_realtime_diarization(sample)
        self.assertIsNone(result.error)
        self.assertEqual(result.segments[0]["speaker"], "Guest-1")
        self.assertIsNone(result.segments[0]["language"])
        self.assertTrue(fake.closed and fake.stopped)

    def test_sdk_lid_passes_candidates_and_records_actual_result(self):
        fake = FakeSpeech()
        original = fake.SpeechRecognizer
        received = {}

        def recognizer(**kwargs):
            received.update(kwargs.pop("auto_detect_source_language_config"))
            return original(**kwargs)

        fake.SpeechRecognizer = recognizer
        fake.languageconfig = SimpleNamespace(AutoDetectSourceLanguageConfig=lambda **kw: kw)
        fake.AutoDetectSourceLanguageResult = lambda result: SimpleNamespace(language="de-DE")
        sample = Sample("s", "test", "en-GB", b"\0" * 6400, "hello world", .2,
                        evaluation_options=EvaluationOptions("multilingual", "at-start", ("en-GB", "de-DE")))
        with patch.object(asr_realtime, "speechsdk", fake), \
                patch.object(asr_realtime.time, "perf_counter", side_effect=lambda: fake.now), \
                patch.object(asr_realtime.time, "sleep"):
            result = asr_realtime.transcribe_realtime(sample)
        self.assertIsNone(result.error)
        self.assertEqual(received["languages"], ["en-GB", "de-DE"])
        self.assertFalse(hasattr(fake.config, "speech_recognition_language"))
        self.assertEqual(result.segments[0]["language"], "de-DE")

    def test_malformed_annotations_become_preserved_failure(self):
        sample = Sample("s", "test", "en-GB", b"\0" * 6400, "hello", .2,
                        evaluation_options=EvaluationOptions(diarization=True))
        result = AsrResult("fast_default", "hello", 10, 20,
                           segments=[segment("hello", "a", start=0, end=float("nan"))])
        entry = evaluation_record(sample, result)
        self.assertIn("invalid_annotation", result.error)
        self.assertEqual(entry["segments"], [])
        self.assertIsNotNone(entry["raw_annotation_evidence"])
        self.assertEqual(entry["lid"]["status"], "service_error")


class ManifestAndRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "audio.wav"
        write_pcm16_wav(b"\0" * 32000, self.audio)
        self.manifest = self.root / "manifest.jsonl"
        self.entry = {"schema_version": 1, "sample_id": "s", "audio": "audio.wav",
                      "requested_language": "en-GB", "scenario": "single-language"}

    def load(self, reference=None):
        entry = dict(self.entry)
        if reference is not None:
            entry["reference"] = reference
        self.manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        return datasets_loader.load_manifest(self.manifest)

    def test_manifest_no_reference_and_require_wder(self):
        samples = self.load()
        self.assertFalse(samples[0].reference_available)
        with self.assertRaisesRegex(ValueError, "speaker-labeled"):
            datasets_loader.prepare_evaluation(samples, ["fast_default"],
                                               EvaluationOptions(diarization=True, score_wder=True))
        datasets_loader.prepare_evaluation(samples, ["fast_default"], EvaluationOptions(diarization=True))

    def test_manifest_invalid_annotations_and_duplicates(self):
        for references in ({"segments": [{"text": "a", "start_s": .8, "end_s": .3}]},
                           {"segments": [{"text": "a", "speaker": {}}]},
                           {"segments": [{"text": "a"}], "text": "different"}):
            with self.subTest(reference=references), self.assertRaises(ValueError):
                self.load(references)
        self.load()
        line = self.manifest.read_text()
        self.manifest.write_text(line + line)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            datasets_loader.load_manifest(self.manifest)

    def test_scored_overlap_rejected(self):
        samples = self.load({"segments": [segment("a", "x", start=0, end=.7),
                                          segment("b", "y", start=.5, end=1)]})
        with self.assertRaisesRegex(ValueError, "overlap"):
            datasets_loader.prepare_evaluation(samples, ["fast_default"],
                                               EvaluationOptions(diarization=True, score_wder=True))

    def run_fake(self, *, scored=True):
        reference = {"segments": [segment("one two", "a"), segment("three four", "b")]} if scored else None
        samples = self.load(reference)
        options = EvaluationOptions(diarization=True, score_wder=scored)
        path = self.root / "run.csv"
        metadata = make_metadata(samples, ["fast_default"], pace=True, seed=42, limit=0, workers=1,
                                 phrases=[], phrase_weight=1, evaluation_options=options)
        metadata_path(path).write_text(json.dumps(metadata), encoding="utf-8")
        result = AsrResult("fast_default", "one two three four", 10, 20, request_id="r",
                           segments=[segment("one two", "x"), segment("three", "y"), segment("four", "x")])
        adapter = Mock(return_value=result)
        with patch.dict(runner.ADAPTERS, {"fast_default": adapter}):
            runner.run_benchmark(samples, path, services=("fast_default",), workers=1,
                                 retries=0, evaluation_options=options)
            check_resume(path, metadata)
            runner.run_benchmark(samples, path, services=("fast_default",), workers=1,
                                 retries=0, evaluation_options=options)
        self.assertEqual(adapter.call_count, 1)
        build_report(path, path.with_name("run_report.md"))
        error_analysis.main(path, path.with_name("run_error_analysis.md"))
        self.assertTrue(validate_artifacts(path)["ok"])
        return path, samples, metadata

    def test_full_scored_run_resume_and_tamper_detection(self):
        path, samples, metadata = self.run_fake()
        entry = json.loads(evaluation_path(path).read_text())
        self.assertEqual(entry["wder"]["rate"], .25)
        self.assertEqual(len(metadata["expected_keys"][0]), 4)
        changed = EvaluationOptions("multilingual", "native", diarization=True, score_wder=True)
        different = make_metadata(samples, ["fast_default"], pace=True, seed=42, limit=0, workers=1,
                                  phrases=[], phrase_weight=1, evaluation_options=changed)
        with self.assertRaisesRegex(ValueError, "differs"):
            check_resume(path, different)
        with self.assertRaisesRegex(ValueError, "strict token"):
            loose_match.main(path)
        entry["wder"]["rate"] = 0
        evaluation_path(path).write_text(json.dumps(entry) + "\n")
        self.assertFalse(validate_artifacts(path)["ok"])

    def test_no_reference_has_no_scores(self):
        path, _, _ = self.run_fake(scored=False)
        with path.open(newline="") as source:
            row = next(csv.DictReader(source))
        self.assertEqual(row["wer"], "")
        entry = json.loads(evaluation_path(path).read_text())
        self.assertIsNone(entry["reference"])
        self.assertIsNone(entry["wder"])
        report = path.with_name("run_error_analysis.md").read_text()
        self.assertIn("[not provided]", report)
        self.assertNotIn("Best service", report)

    def test_cli_rejects_before_adapter_calls(self):
        self.load()
        argv = ["run_full.py", "--manifest", str(self.manifest), "--services", "realtime_mai_2",
                "--diarization", "--output-dir", str(self.root / "out")]
        with patch("sys.argv", argv), patch.object(run_full, "assert_credentials"), \
                patch.object(run_full, "run_benchmark") as execute, redirect_stdout(io.StringIO()), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_full.main()
        execute.assert_not_called()

    def test_variant_changes_with_requested_language(self):
        sample = self.load()[0]
        sample.evaluation_options = EvaluationOptions()
        initial = variant_id(sample, "fast_default")
        sample.locale = "de-DE"
        self.assertNotEqual(initial, variant_id(sample, "fast_default"))

    def test_ordinary_monolingual_normalization_still_works(self):
        samples = self.load({"text": "airflow"})
        options = EvaluationOptions()
        path = self.root / "plain.csv"
        metadata = make_metadata(samples, ["fast_default"], pace=True, seed=42, limit=0, workers=1,
                                 phrases=[], phrase_weight=1, evaluation_options=options)
        metadata_path(path).write_text(json.dumps(metadata))
        result = AsrResult("fast_default", "air flow", 1, 1, request_id="r")
        with patch.dict(runner.ADAPTERS, {"fast_default": Mock(return_value=result)}):
            runner.run_benchmark(samples, path, services=("fast_default",), workers=1,
                                 evaluation_options=options)
        loose_match.main(path)
        build_report(path, path.with_name("plain_report.md"))
        error_analysis.main(path, path.with_name("plain_error_analysis.md"))
        self.assertTrue(validate_artifacts(path)["ok"])
        self.assertIn("loose (strict backup preserved)", path.with_name("plain_report.md").read_text())

    def test_manifest_conflicting_cli_selection_is_rejected(self):
        self.load()
        for extra in ("--languages=en-GB", "--max-per-dataset=1"):
            with patch("sys.argv", ["run_full.py", "--manifest", str(self.manifest),
                                    "--services", "fast_default", extra]), \
                    patch.object(run_full, "assert_credentials"), \
                    patch.object(run_full, "run_benchmark") as execute, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                run_full.main()
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
