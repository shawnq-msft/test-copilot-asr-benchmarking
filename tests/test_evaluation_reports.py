import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark import runner
from benchmark.audio_utils import write_pcm16_wav
from benchmark.config import AsrResult, Sample
from benchmark.eval_options import EvaluationOptions
from benchmark.evaluation import build_evaluation_report, build_combined_evaluation_report
from scripts.validate_benchmark import make_metadata, metadata_path


class EvaluationReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_fixture(self, name="run", *, no_reference=False, failed=False):
        audio = self.root / "test audio.wav"
        write_pcm16_wav(b"\0" * 6400, audio)
        samples = [Sample(sid, "set|A", "zh-CN", b"\0" * 6400, text, .2,
                          audio_path=str(audio), reference_available=not no_reference,
                          reference_language="zh-CN")
                   for sid, text in (("a", "甲乙"), ("b", "甲乙丙丁"), ("c", "甲乙丙丁"))]
        options = EvaluationOptions()
        services = ["fast_default", "fast_mai_2"]
        path = self.root / f"{name}.csv"
        metadata = make_metadata(samples, services, pace=True, seed=42, limit=0, workers=1,
                                 phrases=[], phrase_weight=1, evaluation_options=options)
        metadata_path(path).write_text(json.dumps(metadata), encoding="utf-8")

        def result(sample, service):
            hypothesis = {"a": "甲乙", "b": "甲错丙丁", "c": "错错丙丁"}[sample.sample_id]
            if service == "fast_mai_2":
                hypothesis = sample.reference
            return AsrResult(service, hypothesis, 10, -20,
                             error="no_final" if failed else None, request_id="request", session_id="session")

        with patch.dict(runner.ADAPTERS, {
            service: lambda sample, service=service, **_kwargs: result(sample, service) for service in services
        }):
            runner.run_benchmark(samples, path, services=tuple(services), workers=1, retries=0,
                                 evaluation_options=options)
        return path

    def test_populated_analysis_has_exact_cases_metrics_links_and_patterns(self):
        path = self.run_fixture()
        output = self.root / "report.md"
        build_evaluation_report(path, output, details=True)
        text = output.read_text(encoding="utf-8")
        self.assertIn("| best | set&#124;A | a | fast_default", text)
        self.assertIn("| median | set&#124;A | b | fast_default", text)
        self.assertIn("| worst | set&#124;A | c | fast_default", text)
        self.assertIn("0.3000 / 0.2500", text)
        self.assertIn("| lbl_ms | 3 | -20.0000 | -20.0000 | -20.0000 |", text)
        self.assertIn("[audio](test%20audio.wav)", text)
        self.assertIn("| replace |", text)
        self.assertIn("Negative LBL means", text)
        self.assertIn("| set&#124;A | 0.5000 | c |", text)

    def test_failed_and_reference_free_reports_do_not_invent_scores(self):
        for no_reference, failed in ((True, False), (False, True)):
            path = self.run_fixture(f"run-{no_reference}", no_reference=no_reference, failed=failed)
            output = self.root / f"{path.stem}.md"
            build_evaluation_report(path, output, details=True)
            text = output.read_text(encoding="utf-8")
            self.assertIn("no scorable successful rows", text)
            self.assertNotIn("| best |", text)
            if no_reference:
                self.assertIn("Reference-free comparisons", text)
                self.assertIn("[not provided]", text)
            else:
                self.assertIn("Original service failures: 6", text)
                self.assertIn("no_final", text)

    def test_retry_excluded_and_duplicates_rejected(self):
        main = self.run_fixture()
        retry = self.run_fixture("retry", failed=True)
        output = self.root / "summary.md"
        build_combined_evaluation_report([main], output, retry_csv_paths=(retry,))
        text = output.read_text(encoding="utf-8")
        self.assertIn("Original service failures: 0", text)
        self.assertIn("Separate retry (excluded from all main tables)", text)
        self.assertIn("error=no_final", text)
        with self.assertRaisesRegex(ValueError, "distinct"):
            build_combined_evaluation_report([main, main], output)

    def test_missing_sidecar_rows_raise_instead_of_hiding_evidence(self):
        path = self.run_fixture()
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fields = reader.fieldnames
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[:1])
        with self.assertRaisesRegex(ValueError, "identities"):
            build_evaluation_report(path, self.root / "broken.md")
