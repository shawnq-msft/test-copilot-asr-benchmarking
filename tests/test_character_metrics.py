import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from benchmark.config import AsrResult, Sample
from benchmark.eval_options import EvaluationOptions, validate_options
from benchmark.metrics import cer_breakdown, normalize_characters
from benchmark.runner import _to_row
from benchmark import runner
from benchmark.evaluation import build_evaluation_report, evaluation_record
from scripts.validate_benchmark import make_metadata, metadata_path, validate_artifacts
from scripts import loose_match


class CharacterMetricTests(unittest.TestCase):
    def test_exact_chinese_edit_counts(self):
        scores = cer_breakdown("打开车窗", "打开天窗")
        self.assertEqual(scores, {"cer": .25, "char_ins": 0, "char_del": 0,
                                  "char_sub": 1, "ref_chars": 4})
        self.assertEqual(cer_breakdown("开窗", "打开窗")["char_ins"], 1)
        self.assertEqual(cer_breakdown("打开窗", "开窗")["char_del"], 1)

    def test_normalization_empty_and_mixed_script(self):
        self.assertEqual(normalize_characters("Ａ Tesla， 打 开！"), "atesla打开")
        self.assertEqual(cer_breakdown("打开，车窗！", "打 开 车 窗")["cer"], 0)
        self.assertEqual(cer_breakdown("", "")["cer"], 0)
        self.assertEqual(cer_breakdown("", "开")["cer"], 1)
        self.assertEqual(cer_breakdown("开", "")["cer"], 1)

    def test_chinese_row_ser_and_unavailable_reference(self):
        sample = Sample("s", "zh-CN_commands", "zh-CN", b"\0" * 6400, "打开车窗", .2,
                        evaluation_options=EvaluationOptions())
        result = AsrResult("fast_default", "打 开 车 窗", 1, 1)
        row = _to_row(sample, result)
        self.assertEqual(row["cer"], 0)
        self.assertEqual(row["sentence_error"], 0)
        sample.reference_available = False
        self.assertEqual(_to_row(sample, result)["cer"], "")
        self.assertEqual(_to_row(sample, result)["sentence_error"], "")
        sample.reference_available = True
        result.error = "service failure"
        self.assertEqual(_to_row(sample, result)["cer"], "")

    def test_zh_cn_preflight_is_explicitly_supported(self):
        validate_options(EvaluationOptions(), ["fast_default", "fast_llm", "fast_mai_1_5",
                                               "fast_mai_2", "realtime", "realtime_refine"], ["zh-CN"])

    def test_chinese_segment_whitespace_is_not_annotation_failure(self):
        sample = Sample("s", "zh", "zh-CN", b"\0" * 6400, "打开车窗", .2,
                        evaluation_options=EvaluationOptions())
        result = AsrResult("fast_default", "打开车窗", 1, 1,
                           segments=[{"text": "打开"}, {"text": "车窗"}])
        self.assertIsNone(evaluation_record(sample, result)["error"])

    def test_character_report_and_tamper_validation(self):
        options = EvaluationOptions()
        samples = [Sample("a", "zh", "zh-CN", b"\0" * 6400, "打开车窗", .2),
                   Sample("b", "zh", "zh-CN", b"\0" * 6400, "开窗", .2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            metadata = make_metadata(samples, ["fast_default"], pace=True, seed=42, limit=0,
                                     workers=1, phrases=[], phrase_weight=1,
                                     evaluation_options=options)
            self.assertEqual(metadata["evaluation"]["primary_metric"], "cer")
            metadata_path(path).write_text(json.dumps(metadata), encoding="utf-8")
            adapter = Mock(side_effect=[AsrResult("fast_default", "打开天窗", 1, 1, request_id="a"),
                                        AsrResult("fast_default", "开窗", 1, 1, request_id="b")])
            with patch.dict(runner.ADAPTERS, {"fast_default": adapter}):
                runner.run_benchmark(samples, path, services=("fast_default",), workers=1,
                                     retries=0, evaluation_options=options)
            self.assertTrue(validate_artifacts(path, require_reports=False)["ok"])
            report = path.with_name("run_report.md")
            build_evaluation_report(path, report)
            text = report.read_text(encoding="utf-8")
            self.assertIn("0.1667 / 0.1250", text)
            self.assertIn("Primary accuracy metric: `cer`", text)
            self.assertIn("Mean WER (whitespace)", text)
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            rows[0]["cer"] = "0"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            self.assertFalse(validate_artifacts(path, require_reports=False)["ok"])

    def test_loose_word_matching_preserves_chinese_character_scores(self):
        sample = Sample("s", "zh", "zh-CN", b"\0" * 6400, "30 degrees", .2)
        row = _to_row(sample, AsrResult("fast_default", "30 grad", 1, 1))
        self.assertEqual(row["sentence_error"], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            loose_match.main(path)
            with path.open(encoding="utf-8", newline="") as stream:
                actual = next(csv.DictReader(stream))
            self.assertEqual(float(actual["wer"]), 0)
            self.assertEqual(actual["sentence_error"], "1")
            self.assertEqual(actual["cer"], str(row["cer"]))
            self.assertTrue(path.with_suffix(".csv.strict").exists())


if __name__ == "__main__":
    unittest.main()
