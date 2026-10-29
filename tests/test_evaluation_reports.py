import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark import runner
from benchmark.audio_utils import write_pcm16_wav
from benchmark.config import AsrResult, Sample
from benchmark.eval_options import EvaluationOptions
from benchmark.evaluation import build_evaluation_report, build_combined_evaluation_report
from benchmark.evaluation_report import _link
from benchmark.report import build_report
from scripts.validate_benchmark import make_metadata, metadata_path


class EvaluationReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @unittest.skipUnless(os.name == "nt", "Windows cross-drive paths")
    def test_cross_drive_artifact_link(self):
        source = self.root / "manifest with spaces.jsonl"
        source.write_text("{}\n")
        other_drive = "D:" if source.drive.upper() != "D:" else "C:"
        output = Path(other_drive + r"\reports\report.md")
        self.assertEqual(_link(source, output, "manifest"), f"[manifest]({source.as_uri()})")

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

    def test_classic_template_is_shared_with_legacy_and_keeps_tag(self):
        path = self.run_fixture()
        output = self.root / "report.md"
        inputs = {p: p.read_bytes() for p in self.root.iterdir() if p != output}
        with patch("benchmark.report.tcp_ping_ms") as ping, \
                patch("benchmark.report.public_ip_and_location") as location:
            build_report(path, output, tag="fixed-template")
            first = output.read_text(encoding="utf-8")
            build_report(path, output, tag="fixed-template")
            self.assertEqual(first, output.read_text(encoding="utf-8"))
            ping.assert_not_called()
            location.assert_not_called()
        for source, content in inputs.items():
            self.assertEqual(content, source.read_bytes())
        self.assertTrue(first.startswith("# ASR Benchmark — fixed-template\n"))
        self.assertIn("Report template: **classic-v1**", first)
        expected_sections = [
            "## Endpoints under test", "## Datasets under test", "## Results",
            "## Speech boundaries", "## Latency definitions (all values in ms)",
        ]
        sections = [line for line in first.splitlines() if line.startswith("## ")]
        self.assertEqual(sections[:5], expected_sections)
        self.assertEqual(sections[5], "## Evaluation appendix")
        header = ("| Dataset | Service | N | Errors | WER | SER | INS/100 | DEL/100 | SUB/100 | "
                  "First Latency ms (mean / p90) | LBL ms (mean / p90) | UPL ms (mean / p90) |")
        self.assertIn(header, first)
        self.assertIn("| set&#124;A | fast_default &#91;", first)
        self.assertIn("| - | -20 / -20 | - |", first)
        self.assertIn("| lbl_ms | 3 | -20.0000 | -20.0000 | -20.0000 |", first)
        timeline = first.split("```\n", 2)[1]
        self.assertEqual(timeline, """  audio_start         speech_start                  speech_end       end_of_audio
       │                   │                              │                │
       ├───────────────────┼──────────────────────────────┼────────────────┤
       │  leading silence  │        user speaking         │trailing silence│
       │                   │                              │                │
       │                   │   first_recognizing event    │                │   last_recognized event
       │                   │             ▼                │                │           ▼
       │                   ├────────────→│ First Latency  │                │           │
       │                   │                              │                │           │
       │                   │                              ├────────────────┼──────────→│ UPL
       │                   │                              │                │           │
       │                   │                              │                ├──────────→│ LBL  (can be negative)
""")

        path.with_name(path.stem + "_evaluation.jsonl").unlink()
        with patch("benchmark.report.tcp_ping_ms", return_value=None), \
                patch("benchmark.report.public_ip_and_location", return_value={}):
            build_report(path, output, tag="fixed-template")
        legacy = output.read_text(encoding="utf-8")
        self.assertEqual([line for line in legacy.splitlines() if line.startswith("## ")],
                         expected_sections)
        self.assertIn(header, legacy)
        self.assertEqual(legacy.split("```\n", 2)[1], timeline)

    def test_classic_reference_free_and_failed_headlines_keep_unavailable_values(self):
        for no_reference, failed in ((True, False), (False, True)):
            path = self.run_fixture(f"classic-{no_reference}", no_reference=no_reference, failed=failed)
            output = self.root / f"{path.stem}.md"
            build_report(path, output)
            text = output.read_text(encoding="utf-8")
            summary = text.split("## Results\n", 1)[1].split("## Speech boundaries\n", 1)[0]
            rows = [line for line in summary.splitlines() if line.startswith("| set&#124;A |")]
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertIn(f"| 3 | {3 if failed else 0} | - | - | - | - | - | - |", row)
            self.assertIn("no scorable successful rows", text)
            self.assertNotIn("| best |", text)
            if failed:
                self.assertIn("Original service failures: 6", text)
            else:
                self.assertIn("Reference-free comparisons", text)

    def test_classic_does_not_pool_variants(self):
        path = self.run_fixture()
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fields = reader.fieldnames
        rows[0]["variant_id"] = "separate-variant"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        sidecar = path.with_name(path.stem + "_evaluation.jsonl")
        entries = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
        for entry in entries:
            if entry["sample_id"] == rows[0]["sample_id"] and entry["service"] == rows[0]["service"]:
                entry["variant_id"] = rows[0]["variant_id"]
        sidecar.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
        output = self.root / "variants.md"
        build_report(path, output)
        summary = output.read_text(encoding="utf-8").split("## Results\n", 1)[1].split("## Speech boundaries\n", 1)[0]
        rows = [line for line in summary.splitlines() if line.startswith("| set&#124;A |")]
        self.assertEqual(len(rows), 3)
        self.assertTrue(any("separate-variant&#93; | 1 |" in row for row in rows))

    def test_regeneration_preserves_recorded_recovery_policy(self):
        path = self.run_fixture()
        meta = metadata_path(path)
        metadata = json.loads(meta.read_text(encoding="utf-8"))
        metadata["experiment"] = {
            "kind": "derived_recovery_view_not_a_new_execution",
            "latency_policy": "Descriptive mixed-attempt timings only; not evidence of speedup.",
        }
        meta.write_text(json.dumps(metadata), encoding="utf-8")
        for details in (False, True):
            output = self.root / f"recovery-{details}.md"
            build_evaluation_report(path, output, details=details)
            text = output.read_text(encoding="utf-8")
            self.assertIn("derived_recovery_view_not_a_new_execution", text)
            self.assertIn("not evidence of speedup", text)
