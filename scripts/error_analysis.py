"""Error analysis companion to the benchmark report.

Produces per-(dataset, service) best/median/worst WER, fast_llm hallucination
list, top cross-service disagreements, and [▶] audio links for manual review.

Usage:
    python -X utf8 scripts/error_analysis.py results/<run>.csv results/<run>_error_analysis.md
    # Or called automatically from run_full.py after each benchmark run.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.datasets_loader import iter_samples, write_pcm16_wav  # noqa: E402
from benchmark.metrics import aggregate, normalize_text  # noqa: E402
from benchmark.config import SERVICES as ALL_KNOWN_SERVICES, STREAMING_SERVICES  # noqa: E402



def wer_calc(ref: str, hyp: str) -> float:
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return min(Levenshtein.distance(ref_words, hyp_words) / len(ref_words), 1.0)


def _audio_link(out_dir: Path, ds: str, sid: str,
                cache: dict[tuple[str, str], Path | None],
                samples_by_key: dict[tuple[str, str], object]) -> str:
    key = (ds, sid)
    if key in cache:
        path = cache[key]
    else:
        sample = samples_by_key.get(key)
        if sample is None:
            cache[key] = None
            return ""
        sample_stem = sid[:-4] if sid.lower().endswith(".wav") else sid
        path = out_dir / "audio" / ds / f"{sample_stem}.wav"
        if not path.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                write_pcm16_wav(sample.pcm16_mono_16k, path)
            except Exception as e:
                print(f"[wav] failed to write {path}: {e}")
                cache[key] = None
                return ""
        cache[key] = path
    if path is None:
        return ""
    rel = path.relative_to(out_dir).as_posix()
    rel = quote(rel, safe="/:@")
    return f" [▶]({rel})"


def _load_sample_index(needed: set[tuple[str, str]]) -> dict[tuple[str, str], object]:
    out: dict[tuple[str, str], object] = {}
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for ds, sid in needed:
        by_dataset[ds].add(sid)
    for ds, ids in by_dataset.items():
        if not ids:
            continue
        remaining = set(ids)
        try:
            for s in iter_samples(ds, limit=None):
                if s.sample_id in remaining:
                    out[(ds, s.sample_id)] = s
                    remaining.discard(s.sample_id)
                    if not remaining:
                        break
        except Exception as e:
            print(f"[load] {ds}: {e}")
        if remaining:
            print(f"[load] {ds}: did not find {len(remaining)} requested sample ids")
    return out


def _agg_breakdown(ok_rows: list[dict]) -> tuple[str, str, str]:
    ins = sum(int(r["ins"]) for r in ok_rows if r.get("ins") not in ("", None))
    dele = sum(int(r["del"]) for r in ok_rows if r.get("del") not in ("", None))
    sub = sum(int(r["sub"]) for r in ok_rows if r.get("sub") not in ("", None))
    ref_total = sum(int(r["ref_len"]) for r in ok_rows if r.get("ref_len") not in ("", None))
    if ref_total == 0:
        return "-", "-", "-"
    return (f"{100.0 * ins / ref_total:.1f}",
            f"{100.0 * dele / ref_total:.1f}",
            f"{100.0 * sub / ref_total:.1f}")


def main(csv_path: Path, out_path: Path) -> None:
    from benchmark.evaluation import evaluation_path, build_evaluation_report
    if evaluation_path(csv_path).exists():
        build_evaluation_report(csv_path, out_path, details=True)
        return
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    services_in_csv = sorted({r["service"] for r in rows})
    SERVICES = tuple(s for s in ALL_KNOWN_SERVICES if s in services_in_csv)
    SERVICES += tuple(s for s in services_in_csv if s not in SERVICES)

    by_sample: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_sample[(r["dataset"], r["sample_id"])][r["service"]] = r

    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_key[(r["dataset"], r["service"])].append(r)

    needed: set[tuple[str, str]] = set()

    WORST_N = 10
    worst_rows: list[dict] = []
    for r in rows:
        if r["error"] or r["wer"] == "":
            continue
        if float(r["wer"]) > 0:
            worst_rows.append(r)
    worst_rows.sort(key=lambda r: float(r["wer"]), reverse=True)
    worst_rows = worst_rows[:WORST_N]
    for r in worst_rows:
        needed.add((r["dataset"], r["sample_id"]))

    sub_patterns: Counter = Counter()
    for r in rows:
        if r["error"] or r["wer"] == "":
            continue
        ref_norm = normalize_text(r.get("reference", "") or "")
        hyp_norm = normalize_text(r.get("hypothesis", "") or "")
        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()
        if len(ref_words) == len(hyp_words):
            for rw, hw in zip(ref_words, hyp_words):
                if rw != hw:
                    sub_patterns[(rw, hw)] += 1

    all_keys = set(by_sample.keys())
    top_diffs: list = []
    for key in all_keys:
        ds, sid = key
        svcmap = by_sample[key]
        scored = [(min(float(row["wer"]), 1.0), service)
                  for service, row in svcmap.items() if not row["error"] and row["wer"] != ""]
        if len(scored) >= 2:
            best, worst = min(scored), max(scored)
            top_diffs.append((worst[0] - best[0], ds, sid, svcmap, best, worst))
    top_diffs.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    for _, ds, sid, _, _, _ in top_diffs[:10]:
        needed.add((ds, sid))

    hallucinations: list[tuple[str, str, str, str, str, str]] = []
    for r in rows:
        if r["service"] != "fast_llm" or r.get("error") or r["wer"] == "":
            continue
        wer_val = float(r["wer"])
        if wer_val < 0.8:
            continue
        ref = r.get("reference", "") or ""
        hyp = r.get("hypothesis", "") or ""
        if not ref or not hyp:
            continue
        ref_norm = normalize_text(ref)
        hyp_norm = normalize_text(hyp)
        ref_words = set(ref_norm.split())
        hyp_words = set(hyp_norm.split())
        overlap = len(ref_words & hyp_words)
        if overlap <= 1:
            hallucinations.append((
                r["dataset"], r["sample_id"], ref, hyp,
                r.get("boundary_fix_action", "") or "-",
                f"{wer_val:.3f}",
            ))
            needed.add((r["dataset"], r["sample_id"]))

    print(f"[load] indexing {len(needed)} samples for audio links ...")
    samples_by_key = _load_sample_index(needed)
    print(f"[load] resolved {len(samples_by_key)} samples")
    out_dir = out_path.parent
    audio_cache: dict[tuple[str, str], Path | None] = {}
    datasets_seen = sorted({r["dataset"] for r in rows})

    def metrics_for(rs: list[dict]) -> dict:
        ok = [r for r in rs if not r["error"]]
        wers = [min(float(r["wer"]), 1.0) for r in ok if r["wer"] != ""]
        sers = [int(r["sentence_error"]) for r in ok if r["sentence_error"] != ""]
        ll = [float(r["lbl_ms"]) for r in ok if r["lbl_ms"] not in ("", None)]
        upl = [float(r["upl_ms"]) for r in ok if r.get("upl_ms") not in ("", None)]
        fl = [float(r["first_latency_ms"]) for r in ok
              if r["service"] in STREAMING_SERVICES and r.get("first_latency_ms") not in ("", None)]
        ins_rate, del_rate, sub_rate = _agg_breakdown(ok)
        return {
            "n": len(rs),
            "errs": sum(1 for r in rs if r["error"]),
            "wer": (sum(wers) / len(wers)) if wers else None,
            "ser": (sum(sers) / len(sers)) if sers else None,
            "ins": ins_rate, "del": del_rate, "sub": sub_rate,
            "lbl": aggregate(ll),
            "upl": aggregate(upl),
            "first": aggregate(fl),
        }

    lines: list[str] = []
    lines.append(f"# Error analysis — {csv_path.name}\n")
    lines.append("Audio links (▶) point to `results/audio/<dataset>/<sample_id>.wav` so a reviewer can play the clip directly.")
    lines.append("")

    lines.append("## Datasets under test")
    lines.append("")
    for ds in datasets_seen:
        n_samples = len({r["sample_id"] for r in rows if r["dataset"] == ds})
        parts = ds.rsplit("_", 1)
        loc_name = parts[0] if len(parts) == 2 else ds
        scn = parts[1] if len(parts) == 2 else ""
        lines.append(f"- **{ds}** — {loc_name} {scn} ({n_samples} samples)")
    lines.append("")

    total_samples = len(by_sample)
    lines.append(f"Total samples: **{total_samples}**  ")
    lines.append("")

    realtime_rows = [r for r in rows if r["service"] == "realtime"]
    fixed_rows = [r for r in realtime_rows
                  if r.get("boundary_fix_action") and r["boundary_fix_action"] != "none"]
    lines.append("## Speech boundaries")
    lines.append("")
    lines.append("Shared `speech_start_s` / `speech_end_s` come from baseline realtime when available. "
                 "Check each row's `upl_anchor`; self or missing anchors do not support shared-anchor comparisons. "
                 "Check the report/run metadata for pacing before interpreting latency.")
    lines.append("")
    if fixed_rows:
        action_counts = Counter(r["boundary_fix_action"] for r in fixed_rows)
        lines.append(f"Boundary-fix actions across {len(realtime_rows)} realtime samples: " +
                     ", ".join(f"`{a}`={n}" for a, n in sorted(action_counts.items())))
        lines.append("")
    elif realtime_rows:
        lines.append("All realtime samples passed the boundary-fix heuristic without trimming.")
        lines.append("")
    else:
        lines.append("No baseline realtime boundary checks were available.")

    lines.append("## Results")
    lines.append("")
    lines.append("INS/DEL/SUB are *rates per 100 reference words*. Their sum ≈ WER × 100.")
    lines.append("")
    lines.append("| Dataset | Service | N | Errors | WER | SER | INS/100 | DEL/100 | SUB/100 | First Latency ms (mean / p90) | LBL ms (mean / p90) | UPL ms (mean / p90) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for ds in datasets_seen:
        for svc in SERVICES:
            rs = by_key.get((ds, svc), [])
            m = metrics_for(rs)
            wer_s = f"{m['wer']:.3f}" if m["wer"] is not None else "-"
            ser_s = f"{m['ser']:.3f}" if m["ser"] is not None else "-"
            lbl = m["lbl"]; upl = m["upl"]
            lbl_s = f"{lbl['mean']:.0f} / {lbl['p90']:.0f}" if lbl["n"] else "-"
            upl_s = f"{upl['mean']:.0f} / {upl['p90']:.0f}" if upl["n"] else "-"
            first = m["first"]
            first_s = f"{first['mean']:.0f} / {first['p90']:.0f}" if first["n"] else "-"
            lines.append(f"| {ds} | {svc} | {m['n']} | {m['errs']} | {wer_s} | {ser_s} | {m['ins']} | {m['del']} | {m['sub']} | {first_s} | {lbl_s} | {upl_s} |")
    lines.append("")
    lines.append("## Best / median / worst by dataset and service\n")
    for (dataset, service), group in sorted(by_key.items()):
        scored_rows = sorted((row for row in group if not row["error"] and row["wer"] != ""),
                             key=lambda row: (float(row["wer"]), row["sample_id"]))
        if scored_rows:
            lines.append(f"### {dataset} / {service}")
            for label, row in (("Best", scored_rows[0]), ("Median", scored_rows[len(scored_rows) // 2]),
                               ("Worst", scored_rows[-1])):
                lines.append(f"- {label}: `{row['sample_id']}` WER={row['wer']}; "
                             f"ref: `{row['reference']}`; hyp: `{row['hypothesis']}`")
            lines.append("")

    lines.append("## Worst errors\n")
    lines.append(f"Top {len(worst_rows)} highest-WER rows across all services:\n")
    lines.append("| Audio | Dataset | Sample | Service | WER | Reference | Hypothesis |")
    lines.append("|---|---|---|---|---:|---|---|")
    for r in worst_rows:
        link = _audio_link(out_dir, r["dataset"], r["sample_id"],
                           audio_cache, samples_by_key).strip() or "-"
        lines.append(f"| {link} | {r['dataset']} | {r['sample_id']} | {r['service']} | {float(r['wer']):.3f} | `{r.get('reference','')}` | `{r.get('hypothesis','')}` |")
    lines.append("")

    lines.append("## Most common substitution patterns\n")
    top_subs = sub_patterns.most_common(15)
    if top_subs:
        lines.append("Equal-length ref/hyp word-level substitutions (across all services):\n")
        lines.append("| Count | Reference word | Hypothesis word |")
        lines.append("|---:|---|---|")
        for (rw, hw), cnt in top_subs:
            lines.append(f"| {cnt} | `{rw}` | `{hw}` |")
        lines.append("")
    else:
        lines.append("_(no equal-length substitution patterns found)_\n")

    lines.append("## fast_llm hallucinations\n")
    lines.append("`fast_llm` does not set a locale — it relies on auto-detection. "
                 "When the acoustic signal is weak or the language is unexpected, it may "
                 "produce text in the wrong language or fabricate content.\n")
    if not hallucinations:
        lines.append("_(none detected)_\n")
    else:
        lines.append(f"Found **{len(hallucinations)}** likely hallucinations "
                     f"(WER ≥ 0.8 and ≤ 1 word overlap with reference):\n")
        lines.append("| Audio | Dataset | Sample | WER | Boundary | Reference | Hypothesis |")
        lines.append("|---|---|---|---:|---|---|---|")
        for ds, sid, ref, hyp, fix, wer_s in sorted(hallucinations):
            link = _audio_link(out_dir, ds, sid, audio_cache, samples_by_key).strip() or "-"
            lines.append(f"| {link} | {ds} | {sid} | {wer_s} | {fix} | `{ref}` | `{hyp}` |")
        lines.append("")

    lines.append("## Top cross-service disagreements\n")
    if not top_diffs:
        lines.append("_(none)_\n")
    else:
        for delta, ds, sid, svcmap, best, worst in top_diffs[:10]:
            link = _audio_link(out_dir, ds, sid, audio_cache, samples_by_key)
            rt = svcmap.get("realtime", {})
            ss = rt.get("speech_start_s", "") or "-"
            se = rt.get("speech_end_s", "") or "-"
            fix = rt.get("boundary_fix_action", "") or "-"
            lines.append(f"### {ds}/{sid}{link}  Δwer={delta:.3f}  ({best[1]}={best[0]:.3f}, {worst[1]}={worst[0]:.3f})  speech=[{ss}s, {se}s] fix={fix}")
            lines.append(f"- ref:           `{svcmap[best[1]].get('reference','')}`")
            for s in SERVICES:
                if s in svcmap:
                    lines.append(f"- {s:<14} `{svcmap[s].get('hypothesis','')}`")
            lines.append("")

    lines.append("## Caveats\n")
    lines.append("- **Shared UPL requires a trustworthy baseline realtime anchor.** MAI-only runs may have no speech-anchored metrics. Missing metrics are not zero; inspect `upl_anchor` and `upl_self_ms`.")
    lines.append("- **Voice commands are short utterances** (typically 3-10 words). WER on short references is noisier — a single word error on a 4-word command gives 25% WER. Interpret per-sample WER with this in mind.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    audio_count = sum(1 for v in audio_cache.values() if v is not None)
    print(f"Wrote {out_path}")
    print(f"Total samples: {total_samples}")
    print(f"  audio files emitted/linked: {audio_count}")


if __name__ == "__main__":
    import argparse
    import benchmark.datasets_loader as loader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--testdata-dir", type=Path)
    args = parser.parse_args()
    if args.testdata_dir:
        loader.DATA_ROOT = args.testdata_dir
    main(args.csv_path, args.out_path)
