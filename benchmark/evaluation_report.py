"""Local, reference-aware ASR reports for single and explicitly combined runs."""
from __future__ import annotations

import csv
import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from rapidfuzz.distance import Levenshtein

from .metrics import aggregate, normalize_characters, normalize_text


def _text(value) -> str:
    return (html.escape(str(value), quote=False).replace("|", "&#124;").replace("\n", "<br>")
            .replace("[", "&#91;").replace("]", "&#93;"))


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.4f}"


def _link(path: Path, output: Path, label: str) -> str:
    if not path.is_file():
        return f"{_text(label)} (unavailable)"
    target, parent = path.resolve(), output.parent.resolve()
    if target.anchor.casefold() != parent.anchor.casefold():
        return f"[{_text(label)}]({target.as_uri()})"
    relative = Path(os.path.relpath(target, parent)).as_posix()
    return f"[{_text(label)}]({quote(relative, safe='/.:')})"


def _read_run(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    evidence_path = path.with_name(path.stem + "_evaluation.jsonl")
    entries = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines() if line]
    key = lambda r: (r["dataset"], r["sample_id"], r["service"], r["variant_id"])
    index = {key(e): e for e in entries}
    if len(index) != len(entries) or Counter(map(key, rows)) != Counter(map(key, entries)):
        raise ValueError(f"CSV/evaluation evidence identities disagree: {path}")
    metadata_path = path.with_name(path.stem + "_run.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    samples = {(s["dataset"], s["sample_id"]): s for s in metadata.get("evaluation", {}).get("samples", [])}
    records = [{"row": r, "entry": index[key(r)], "sample": samples.get((r["dataset"], r["sample_id"]), {}),
                "run": path.stem} for r in rows]
    return {"path": path, "metadata": metadata, "records": records}


def _primary(record: dict) -> str:
    return "cer" if record["row"]["locale"].startswith(("zh-", "yue-")) else "wer"


def _scorable(records: list[dict]) -> list[dict]:
    return [r for r in records if not r["row"]["error"] and r["entry"]["reference_available"]
            and r["row"].get(_primary(r)) not in ("", None)]


def _configuration(record: dict) -> str:
    entry = record["entry"]
    options = entry["options"]
    return (f"{options['language_mode']} / {options['lid_strategy'] or '-'} / "
            f"{entry['requested_language'] or 'unset'} / {entry['variant_id']}")


def _group_key(record: dict) -> tuple:
    row = record["row"]
    return record["run"], row["dataset"], row["service"], row["variant_id"]


def _stats(records: list[dict]) -> dict:
    scored = _scorable(records)
    good = [r["row"] for r in records if not r["row"]["error"]]
    char_rows = [r["row"] for r in scored if r["row"].get("cer") not in ("", None)]
    chars = sum(int(r["ref_chars"]) for r in char_rows)
    edits = [sum(int(r[f]) for r in char_rows) for f in ("char_ins", "char_del", "char_sub")]
    return {"calls": len(records), "failures": len(records)-len(good), "scored": len(scored),
            "chars": chars, "edits": edits, "cer_micro": sum(edits)/chars if chars else None,
            "cer_macro": aggregate([float(r["cer"]) for r in char_rows])["mean"],
            "wer": aggregate([float(r["row"]["wer"]) for r in scored])["mean"],
            "ser": aggregate([float(r["row"]["sentence_error"]) for r in scored])["mean"],
            "latency": {f: aggregate([float(r[f]) for r in good if r.get(f) not in ("", None)])
                        for f in ("first_latency_ms", "lbl_ms", "upl_ms")},
            "anchors": dict(Counter(r.get("upl_anchor") or "unavailable" for r in good))}


def _case(record: dict, output: Path) -> list[str]:
    row = record["row"]
    audio = record["sample"].get("audio_path")
    edits = (" / ".join(str(row.get(f, "-")) for f in ("char_ins", "char_del", "char_sub"))
             if _primary(record) == "cer" else " / ".join(row[f] for f in ("ins", "del", "sub")))
    return [_text(row["sample_id"]), _text(row["service"]), _text(_configuration(record)),
            _primary(record), _fmt(float(row[_primary(record)])), edits,
            _text(row["reference"]), _text(row["hypothesis"]),
            _link(Path(audio), output, "audio") if audio else "audio unavailable"]


def _table(lines: list[str], headers: list[str], rows: list[list[str]]) -> None:
    lines.extend(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)])
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")


def build_reports(csv_paths: list[Path], output: Path, *, details: bool = False,
                  retry_csv_paths: tuple[Path, ...] = (), tag: str = "") -> None:
    if not csv_paths or len(set(csv_paths)) != len(csv_paths):
        raise ValueError("Select distinct, explicit evaluation CSVs")
    runs = [_read_run(p) for p in csv_paths]
    records = [r for run in runs for r in run["records"]]
    identities = [(r["run"], *_group_key(r)[1:], r["row"]["sample_id"]) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate run/sample/service identities")
    groups = defaultdict(list)
    for record in records:
        groups[_group_key(record)].append(record)
    lines = ["# ASR evaluation and error analysis" if details else "# ASR evaluation report", "",
             "[Dataset composition](#datasetset-composition) | "
             "[Accuracy](#accuracy-by-set-and-modelconfiguration) | "
             "[Language subsets](#reference-language-subsets) | "
             "[Latency](#latency-definitions-and-statistics) | "
             "[Ranked cases](#best--median--worst-reference-backed-cases) | "
             "[Differences](#cross-service-differences-and-error-patterns) | "
             "[Failures](#failures-warnings-and-separate-retries)", "",
             "## Scope and run provenance", "",
             "This report reads existing results only. It does not rerun recognition or change scores.",
             "Runs/configurations remain separate. Combined reports include only the explicitly selected CSVs; "
             "smoke and separately listed retries are excluded from main aggregates.",
             "Accuracy/latency exclude failed rows; scored and timing observation counts expose differing denominators.",
             "Reference-free results are descriptive: [not provided] is not a zero-error reference.", ""]
    lines.append("Corpus sampling and benchmark normalization change the evaluation scope; these are not official leaderboard scores.")
    inventories = {}
    for run in runs:
        path, meta = run["path"], run["metadata"]
        normalized = path.with_suffix(path.suffix + ".strict").exists()
        expected = {tuple(k) for k in meta.get("expected_keys", [])}
        actual = {(r["row"]["dataset"], r["row"]["sample_id"], r["row"]["service"], r["row"]["variant_id"])
                  for r in run["records"]}
        coverage = (f"expected {len(expected)}, actual {len(actual)}, missing {len(expected-actual)}, "
                    f"unexpected {len(actual-expected)}" if expected else "expected coverage unavailable")
        lines.extend([f"### {_text(path.stem)}",
                      f"- Coverage: {coverage}; failures: {sum(bool(r['row']['error']) for r in run['records'])}.",
                      f"- WER scoring: {'loose (strict backup preserved)' if normalized else 'strict'}. "
                      "CER and Chinese SER retain strict NFKC/lower/punctuation/whitespace normalization.",
                      f"- Primary accuracy metric: `{meta.get('evaluation', {}).get('primary_metric', 'unavailable')}`. "
                      "Whitespace WER is not Chinese word-segmented WER.",
                      f"- SDK: {_text(meta.get('speech_sdk', 'unavailable'))}; region: {_text(meta.get('speech_region', 'unavailable'))}; "
                      f"endpoint: {_text(meta.get('speech_endpoint', 'unavailable'))}; paced: {meta.get('pace', 'unavailable')}; "
                      f"chunk ms: {meta.get('chunk_ms', 'unavailable')}; workers: {meta.get('workers', 'unavailable')}.",
                      f"- Options: {_text(json.dumps(meta.get('evaluation', {}).get('options', {})))}."])
        if meta.get("experiment"):
            lines.append(f"- Recorded experiment/derivation: {_text(json.dumps(meta['experiment']))}.")
        if meta.get("azure_whisper"):
            lines.append(f"- Azure Whisper (not Whisper v3): {_text(json.dumps(meta['azure_whisper']))}. "
                         "First latency is whole-request time to the non-streaming JSON response; "
                         "LBL is measured from the end of paced upload. No partial or word timestamps.")
        if "mai_sdk_language_codes" in meta:
            lines.append(f"- MAI SDK base language codes: {_text(json.dumps(meta['mai_sdk_language_codes']))}; "
                         "empty means explicitly selected native-auto. This is a requested setting, not a detected-language label.")
        for suffix, label in ((".csv", "CSV"), ("_run.json", "metadata"), ("_validation.json", "validation"),
                              ("_words.jsonl", "streaming evidence"), ("_evaluation.jsonl", "evaluation evidence"),
                              ("_report.md", "full report"), ("_error_analysis.md", "error analysis")):
            target = path if suffix == ".csv" else path.with_name(path.stem + suffix)
            lines.append(f"- {_link(target, output, label)}")
        if normalized:
            lines.append(f"- {_link(path.with_suffix(path.suffix + '.strict'), output, 'strict backup')}")
        manifests = {s.get("manifest_path") for s in meta.get("evaluation", {}).get("samples", []) if s.get("manifest_path")}
        for name in sorted(manifests):
            manifest = Path(name)
            lines.append(f"- {_link(manifest, output, 'selection manifest')}")
            inventory = manifest.parent / "sampling_inventory.json"
            if inventory.is_file():
                inventories[inventory] = json.loads(inventory.read_text(encoding="utf-8"))
        lines.append("")
    lines.extend(["## Dataset/set composition", ""])
    full_counts = {}
    for path, inv in inventories.items():
        lines.append(f"- {_link(path, output, 'sampling inventory')}: "
                     f"seed={inv.get('seed', 'unavailable')}; source recordings={inv.get('source_count', inv.get('source_rows', 'unavailable'))}; "
                     f"selected={inv.get('selected_count', inv.get('sample_rows', 'unavailable'))}; "
                     f"policy={_text(inv.get('policy', inv.get('sampling', 'unavailable')))}.")
        lines.append(f"- Source: {_text(inv.get('source', inv.get('source_manifest', 'see inventory')))}; "
                     f"split: {_text(inv.get('split', 'not declared'))}; revision: {_text(inv.get('revision', 'not declared'))}; "
                     f"license: {_text(inv.get('license', 'not declared for supplied corpus'))}.")
        if inv.get("license_url"):
            lines.append(f"- License terms: {_text(inv['license_url'])}")
        if inv.get("excluded_services"):
            lines.append(f"- Exclusions: {_text(json.dumps(inv['excluded_services']))}")
        ref_inventory = path.parent / "reference_inventory.json"
        if ref_inventory.is_file():
            full_counts.update(json.loads(ref_inventory.read_text(encoding="utf-8")).get("groups", {}))
            lines.append(f"- {_link(ref_inventory, output, 'full source inventory')}")
        elif len({r["row"]["dataset"] for r in records}) == 1 and inv.get("source_rows") is not None:
            full_counts[records[0]["row"]["dataset"]] = inv["source_rows"]
    if not inventories:
        lines.append("Source population/seed/allocation inventory unavailable; selected counts below come from run metadata/evidence.")
    lines.append("No implicit model substitution: unimplemented Whisper is not measured unless explicitly selected. "
                 "Counts are recordings, not unique sentences or model calls. Full counts are unavailable where marked '-'.")
    selected = defaultdict(set)
    refs = defaultdict(set)
    for r in records:
        selected[r["row"]["dataset"]].add(r["row"]["sample_id"])
        refs[r["row"]["dataset"]].add(r["entry"].get("reference_language") or "unavailable")
    _table(lines, ["Dataset / set", "Source recordings", "Selected recordings", "Reference languages"],
           [[_text(ds), str(full_counts.get(ds, "-")), str(len(ids)), _text(", ".join(sorted(refs[ds])))]
            for ds, ids in sorted(selected.items())])
    lines.append(f"Selected total: {sum(map(len, selected.values()))}; sets: {len(selected)}. "
                 "Sampling-balanced results are not population-weighted estimates. Source inventories may describe a broader selection than a narrow smoke/retry run.")
    lines.extend(["", "## Accuracy by set and model/configuration", "",
                  "CER micro = total character edits / reference characters; macro = mean per-recording CER. "
                  "SER is the mean sentence-error flag. I/D/S are character edits for CER. Rates are fractions, not percentages.", ""])

    def accuracy_table(scope_groups):
        cells = []
        for label, group in scope_groups:
            s = _stats(group)
            cells.append([_text(label), _text(group[0]["row"]["service"]), _text(_configuration(group[0])),
                          f"{s['calls']} / {s['failures']} / {s['scored']}",
                          f"{_fmt(s['cer_micro'])} / {_fmt(s['cer_macro'])}", str(s["chars"]),
                          " / ".join(map(str, s["edits"])), _fmt(s["wer"]), _fmt(s["ser"])])
        _table(lines, ["Scope / set", "Service", "Mode / strategy / locale / variant", "Calls / failures / scored",
                       "CER micro / macro", "Ref chars", "Char I / D / S", "Mean WER (whitespace)", "Mean SER"], cells)

    accuracy_table([(key[1], group) for key, group in sorted(groups.items())])
    language_groups = defaultdict(list)
    for r in records:
        language_groups[r["run"], r["entry"].get("reference_language") or "unavailable",
                        r["row"]["service"], r["row"]["variant_id"]].append(r)
    lines.extend(["## Reference-language subsets", "",
                  "Reference language is not detected language. Cantonese/yue-CN under requested zh-CN is an "
                  "out-of-language challenge, not fixed-Cantonese evaluation. Native-auto is a different configuration.", ""])
    accuracy_table([(key[1], group) for key, group in sorted(language_groups.items())])
    lines.extend(["## Latency definitions and statistics", "",
                  "- First: streaming first partial relative to a word-start anchor, not necessarily call-to-first-text. "
                  "REST has no partial stream: its first response is a final response, so these are not equivalent UX measures.",
                  "- LBL = last final event time minus actual end-of-audio upload time. Negative LBL means a final "
                  "arrived while remaining audio was still being sent; it is not negative wall time or proof of speedup.",
                  "- UPL = last final minus the speech-end anchor. `realtime` and `self` anchors are not interchangeable. "
                  "No available word timestamps/anchor means unavailable, not zero. Missing native First can coexist with partial events.",
                  "- Tables use successful calls, repository mean/median/p90 definitions and valid counts per metric. "
                  "Unpaced runs and incompatible modes/anchors do not support direct comparative latency claims.",
                  "- Offline regeneration does not measure current network latency. Historical REST event timing "
                  "cannot be independently reconstructed where only aggregate CSV timing was retained.", ""])
    latency_rows = []
    for key, group in sorted(groups.items()):
        s = _stats(group)
        for name, values in s["latency"].items():
            latency_rows.append([_text(key[1]), _text(key[2]), _text(key[3]), name, str(values["n"]),
                                 *[_fmt(values[k]) for k in ("mean", "median", "p90")], _text(json.dumps(s["anchors"]))])
    _table(lines, ["Set", "Service", "Variant", "Metric (ms)", "N", "Mean", "Median", "p90", "UPL anchors"], latency_rows)
    lines.extend(["## Best / median / worst reference-backed cases", "",
                  "Within each set/model/configuration, sort successful scorable rows by primary error rate, then "
                  "sample ID. Median is the lower middle ranked recording (not the arithmetic median score); ties "
                  "are deterministic. A small group may reuse a recording. Failures are listed separately.", ""])
    case_headers = ["Rank", "Set", "Sample", "Service", "Configuration", "Metric", "Score", "I / D / S", "Reference", "Hypothesis", "Audio"]
    cases = []
    for key, group in sorted(groups.items()):
        ranked = sorted(_scorable(group), key=lambda r: (float(r["row"][_primary(r)]), r["row"]["sample_id"]))
        if not ranked:
            lines.append(f"- {_text(key[1])} / {_text(key[2])} / {_text(key[3])}: no scorable successful rows; ranked cases not applicable.")
            continue
        for label, index in (("best", 0), ("median", (len(ranked)-1)//2), ("worst", len(ranked)-1)):
            cases.append([label, _text(key[1]), *_case(ranked[index], output)])
    _table(lines, case_headers, cases)
    lines.extend(["## Cross-service differences and error patterns", "",
                  "Per set, show up to three matched recordings with the largest primary-score spread; ties use sample ID. "
                  "Each selected recording shows every available scorable endpoint. These are observed differences, "
                  "not causal explanations; native and fixed-language modes are explicitly distinguished.", ""])
    by_sample = defaultdict(list)
    for r in records:
        by_sample[r["row"]["dataset"], r["row"]["sample_id"]].append(r)
    comparisons = defaultdict(list)
    incompatible = 0
    for (ds, sid), group in by_sample.items():
        good = _scorable(group)
        if len({g["row"]["service"] for g in good}) < 2:
            continue
        if len({(g["entry"]["pcm_sha256"],
                 (normalize_characters if _primary(g) == "cer" else normalize_text)(g["row"]["reference"]),
                 _primary(g)) for g in good}) != 1:
            incompatible += 1
            continue
        scores = [float(g["row"][_primary(g)]) for g in good]
        if len({(normalize_characters if _primary(g) == "cer" else normalize_text)(g["row"]["hypothesis"])
                for g in good}) > 1:
            comparisons[ds].append((max(scores)-min(scores), sid, good))
    differences = []
    descriptive = []
    for ds in sorted(selected):
        choices = sorted(comparisons[ds], key=lambda x: (-x[0], x[1]))[:3]
        if not choices:
            lines.append(f"- {_text(ds)}: no eligible cross-service disagreement "
                         "(single endpoint, no scored reference, or identical normalized outputs).")
        for spread, _, group in choices:
            for r in sorted(group, key=lambda r: (r["row"]["service"], r["run"])):
                differences.append([_text(ds), _fmt(spread), *_case(r, output)])
        unscored = [(sid, group) for (dataset, sid), group in by_sample.items() if dataset == ds
                    and len({r["row"]["service"] for r in group if not r["row"]["error"]}) > 1
                    and any(not r["entry"]["reference_available"] for r in group)]
        for sid, group in sorted(unscored)[:3]:
            for r in sorted(group, key=lambda item: (item["row"]["service"], item["run"])):
                entry = r["entry"]
                descriptive.append([_text(ds), _text(sid), _text(r["row"]["service"]),
                                    _text(entry["hypothesis"]), _text(entry["description"]["speakers"]),
                                    _text(entry["description"]["detected_languages"])])
    _table(lines, ["Set", "Score spread", "Sample", "Service", "Configuration", "Metric", "Score", "I / D / S",
                   "Reference", "Hypothesis", "Audio"], differences)
    if descriptive:
        lines.append("Reference-free comparisons (first three sample IDs per set): descriptive only, no accuracy or winner.")
        _table(lines, ["Set", "Sample", "Service", "Hypothesis", "Returned speakers", "Returned languages"], descriptive)
    lines.append(f"Incompatible input/reference groups excluded from paired comparisons: {incompatible}.")
    lines.append("Strict normalized-text edit patterns below are independent of permitted loose WER. "
                 "Insertion-heavy means inserted units exceed reference units; candidates require listening, not an automatic hallucination diagnosis.")
    patterns = []
    for key, group in sorted(groups.items()):
        counts = Counter()
        heavy = []
        for r in _scorable(group):
            row = r["row"]
            normalize = normalize_characters if _primary(r) == "cer" else lambda s: normalize_text(s).split()
            ref, hyp = normalize(row["reference"]), normalize(row["hypothesis"])
            ops = Levenshtein.editops(ref, hyp)
            for op in ops:
                a = str(ref[op.src_pos]) if op.tag != "insert" else "<none>"
                b = str(hyp[op.dest_pos]) if op.tag != "delete" else "<none>"
                counts[op.tag, a, b] += 1
            if sum(op.tag == "insert" for op in ops) > len(ref):
                heavy.append(row["sample_id"])
        for (operation, a, b), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]:
            patterns.append([_text(key[1]), _text(key[2]), _text(key[3]), operation, _text(a), _text(b), str(count)])
        if heavy:
            lines.append(f"- {_text(key[1])} / {_text(key[2])}: insertion-heavy candidates: {_text(', '.join(heavy))}.")
    _table(lines, ["Set", "Service", "Variant", "Operation", "Reference unit", "Hypothesis unit", "Count"], patterns)
    lines.extend(["## Speaker/language evidence (when applicable)", "",
                  "wDER uses one recording-level speaker mapping on aligned correct/substituted words; "
                  "insertions/deletions remain in WER. No time-based DER is computed. Without speaker references, "
                  "speaker agreement is not accuracy. LID uses returned labels only, never configured language.", ""])
    annotation_rows = []
    for key, group in sorted(groups.items()):
        good = [r["entry"] for r in group if not r["row"]["error"]]
        scores = [e["wder"] for e in good if e.get("wder") is not None]
        aligned = sum(s["aligned_words"] for s in scores)
        errors = sum(s["speaker_errors"] for s in scores)
        labels = [e["lid"]["clip_correct"] for e in good if e.get("lid", {}).get("clip_correct") is not None]
        annotation_rows.append([_text(key[1]), _text(key[2]), _text(key[3]),
                                _fmt(errors/aligned if aligned else None),
                                _fmt(aggregate([s["rate"] for s in scores if s["rate"] is not None])["mean"]),
                                str(aligned), _fmt(sum(labels)/len(labels) if labels else None), f"{len(labels)}/{len(good)}"])
    _table(lines, ["Set", "Service", "Variant", "wDER micro", "wDER macro", "Aligned words", "LID accuracy", "Labeled / successful"], annotation_rows)
    lines.extend(["## Failures, warnings and separate retries", "",
                  "No failed row is discarded or replaced. Missing metrics/IDs/audio remain unavailable. "
                  "Detailed final segments, returned labels, transcripts and IDs are retained in linked evaluation/streaming sidecars.", ""])
    failures = []
    for r in records:
        row = r["row"]
        if row["error"]:
            failures.append([_text(row["dataset"]), _text(row["sample_id"]), _text(row["service"]),
                             _text(row["error"]), _text(row["request_id"] or "unavailable"), _text(row["session_id"] or "unavailable")])
    _table(lines, ["Set", "Sample", "Service", "Error", "Request ID", "Session ID"], failures)
    lines.append(f"Original service failures: {len(failures)}.")
    for run in runs:
        path = run["path"].with_name(run["path"].stem + "_validation.json")
        if path.is_file():
            val = json.loads(path.read_text(encoding="utf-8"))
            warnings = Counter(w.split("): ", 1)[-1] for w in val["warnings"])
            lines.append(f"- {_text(run['path'].stem)}: saved validation ok={val['ok']}; "
                         f"structural errors={_text(json.dumps(val['errors']))}; warnings={_text(json.dumps(dict(warnings)))}.")
        else:
            lines.append(f"- {_text(run['path'].stem)}: validation unavailable; do not infer success.")
    for path in retry_csv_paths:
        retry = _read_run(path)
        lines.append(f"- Separate retry (excluded from all main tables): {_link(path, output, 'retry CSV')}; "
                     f"{_link(path.with_name(path.stem+'_report.md'), output, 'retry report')}.")
        for r in retry["records"]:
            row = r["row"]
            lines.append(f"  - {_text(row['sample_id'])} / {_text(row['service'])}: error={_text(row['error'] or 'none')}; "
                         f"CER={row.get('cer') or 'unavailable'}; request={_text(row['request_id'] or 'unavailable')}; "
                         f"session={_text(row['session_id'] or 'unavailable')}.")
    if not retry_csv_paths:
        lines.append("No separate retry CSVs supplied to this report; no retries were merged.")
    lines.extend(["", "## Verification", "",
                  "Report generation is not verification. Independently check these tables, cases and links against "
                  "the linked source artifacts before marking report_verified. Data/service success is a separate fact.", ""])
    for path in csv_paths:
        lines.append(f"- {_link(path.with_name(path.stem + '_verification.json'), output, 'run verification')}")
    if output.name not in {p.stem + suffix for p in csv_paths for suffix in ("_report.md", "_error_analysis.md")}:
        lines.append(f"- {_link(output.with_name(output.stem + '_verification.json'), output, 'combined verification')}")
    if not details:
        from .report import render_report_lines

        headline = []
        for run in runs:
            rows = [dict(r["row"], reference_available=r["entry"]["reference_available"])
                    for r in run["records"]]
            headline.extend(render_report_lines(run["path"], rows, tag=tag,
                                                metadata=run["metadata"], local=True))
            headline.append("")
        lines = headline + ["## Evaluation appendix", ""] + lines[2:]
    output.write_text("\n".join(lines), encoding="utf-8")
