"""Generate the benchmark report markdown from a results CSV."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .config import (AZURE_SPEECH_ENDPOINT, AZURE_SPEECH_REGION,
                     FAST_API_VERSION, LLM_API_VERSION,
                     SEGMENTATION_SILENCE_MS,
                     REFINE_AZURE_ENDPOINT, REFINE_AZURE_REGION,
                     MAI_STREAMING_MODEL, STREAMING_SERVICES,
                     MAI_REALTIME_ENDPOINT, MAI_REALTIME_DEPLOYMENT,
                     llm_rest_base_url, llm_rest_host, rest_base_url, rest_host)
from .metrics import aggregate
from . import config
from .ping import public_ip_and_location, tcp_ping_ms

_HOST = rest_host()
_REST_BASE = rest_base_url()
_LLM_HOST = llm_rest_host()
_LLM_REST_BASE = llm_rest_base_url()


def _endpoints_for_locale(locale: str) -> dict[str, dict]:
    locale_short = locale.split("-")[0]
    return {
        "fast_default": {
            "name": "Azure Fast Transcription (default)",
            "url": f"{_REST_BASE}?api-version={FAST_API_VERSION}",
            "transport": "HTTPS POST (multipart/form-data, chunked)",
            "config": f'definition = {{"locales": ["{locale}"]}}',
            "partials": "no",
            "doc": "https://learn.microsoft.com/en-us/azure/ai-services/speech-service/fast-transcription-create",
        },
        "fast_llm": {
            "name": "Azure Fast Transcription — LLM enhanced (preview)",
            "url": f"{_LLM_REST_BASE}?api-version={LLM_API_VERSION}",
            "transport": "HTTPS POST (multipart/form-data, chunked)",
            "config": 'definition = {"enhancedMode": {"enabled": true, "task": "transcribe"}}',
            "partials": "no",
            "note": "Requires Speech resource with LLM-Speech preview enabled. No locale set — relies on auto-detection.",
            "doc": "https://learn.microsoft.com/en-us/azure/ai-services/speech-service/llm-speech",
        },
        "fast_mai_1_5": {
            "name": "Azure Fast Transcription — MAI 1.5 model (preview)",
            "url": f"{_LLM_REST_BASE}?api-version={LLM_API_VERSION}",
            "transport": "HTTPS POST (multipart/form-data, chunked)",
            "config": f'definition = {{"locales": ["{locale_short}"], "enhancedMode": {{"enabled": true, "model": "MAI-Transcribe-1.5"}}}}',
            "partials": "no",
            "note": "Requires a Foundry Speech resource in a supported region. Phrase list and transcribe style are supported.",
            "doc": "https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe",
        },
        "fast_mai_2": {
            "name": "Azure Fast Transcription — MAI 2 model (preview)",
            "url": f"{_LLM_REST_BASE}?api-version={LLM_API_VERSION}",
            "transport": "HTTPS POST (multipart/form-data, chunked)",
            "config": f'definition = {{"locales": ["{locale_short}"], "enhancedMode": {{"enabled": true, "model": "MAI-Transcribe-2"}}}}',
            "partials": "no",
            "note": "Requires a Foundry Speech resource in a supported region. Phrase list, transcription styles, diarization, and word timestamps are supported by the model.",
            "doc": "https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe",
        },
        "realtime": {
            "name": "Azure Speech SDK — continuous recognition",
            "url": (AZURE_SPEECH_ENDPOINT or
                    f"wss://{AZURE_SPEECH_REGION}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"),
            "transport": "WebSocket via azure-cognitiveservices-speech SDK",
            "config": f'PushAudioInputStream, language="{locale}", continuous',
            "partials": "yes (recognizing/recognized events)",
            "doc": "https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-recognize-speech",
        },
        "realtime_refine": {
            "name": "Azure Speech SDK — continuous + Post-Stream Refinement (MRS preview)",
            "url": (REFINE_AZURE_ENDPOINT or
                    f"wss://{REFINE_AZURE_REGION}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"),
            "transport": "WebSocket via azure-cognitiveservices-speech SDK (>=1.49.0)",
            "config": (f'PushAudioInputStream, language="{locale}", continuous, '
                       'PostProcessingOption="PostRefinement"'),
            "partials": "yes (recognizing/recognized events; final replaced after refinement)",
            "note": ("Requires Speech SDK >= 1.49.0 and a Speech resource in a region where "
                     "Multi-Recognizer / Post-Stream Refinement public preview is enabled. "
                     "Falls back to non-MRS behavior if PostProcessingOption is not set."),
            "doc": "https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-recognize-speech",
        },
        "realtime_mai_2": {
            "name": "MAI-Transcribe-2-Streaming (public preview)",
            "url": AZURE_SPEECH_ENDPOINT or f"SDK region: {AZURE_SPEECH_REGION}",
            "transport": "Azure Speech SDK >=1.52.0, continuous PushAudioInputStream",
            "config": f'SpeechConfig.model="{MAI_STREAMING_MODEL}"',
            "partials": "yes (recognizing/recognized events)",
            "note": ("Monolingual runs set a base language code, not a full locale; native mode leaves it unset. "
                     "Phrase grammar, segmentation and detailed-word options are not enabled. "
                     "Uses baseline speech boundaries when available; absent word timing is not fabricated."),
            "doc": "https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe-2-streaming-speech-sdk",
        },
        "realtime_mai_2_api": {
            "name": "MAI-Transcribe-2-Streaming via Realtime API (opt-in preview)",
            "url": MAI_REALTIME_ENDPOINT + "/mai/v1/realtime?intent=transcription",
            "transport": "WebSocket, PCM16 base64 audio, explicit end-of-recording commit",
            "config": f"deployment={MAI_REALTIME_DEPLOYMENT}, 16 kHz mono, language auto-detection",
            "partials": "yes (intermediate and finalized delta events)",
            "note": ("Uses the same resource key as SDK but its documented MAI Realtime endpoint and "
                     "a deployed model. First latency is first nonempty intermediate; first-text and "
                     "setup timing are also in the sidecar. No automatic turn detection. "
                     "Do not confuse faster explicit commit with SDK overhead."),
            "doc": "https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe-2-streaming-realtime",
        },
        "whisper_v3": {
            "name": "Azure Fast Transcription — Whisper large-v3 (stub)",
            "url": f"{_REST_BASE}?api-version={FAST_API_VERSION}",
            "transport": "HTTPS POST (multipart/form-data, chunked)",
            "config": f'definition = {{"locales": ["{locale}"], "model": "whisper-large-v3"}}',
            "partials": "no",
            "note": "Not yet implemented — returns not_implemented error.",
            "doc": "https://learn.microsoft.com/en-us/azure/ai-services/speech-service/fast-transcription-create",
        },
        "azure_whisper": {
            "name": "Azure OpenAI Whisper (not Whisper v3)",
            "url": config.WHISPER_AZURE_ENDPOINT,
            "transport": "HTTPS multipart/form-data, paced upload, final JSON response",
            "config": (f"deployment={config.WHISPER_AZURE_DEPLOYMENT}, "
                       f"model=whisper/{config.WHISPER_AZURE_MODEL_VERSION}, language={locale_short}"),
            "partials": "no",
            "note": "Whole-request first latency; LBL from end of upload; no fabricated word timestamps.",
            "doc": "https://learn.microsoft.com/en-us/azure/ai-foundry/openai/whisper-quickstart",
        },
    }


def build_report(csv_path: Path, md_path: Path, tag: str = "") -> None:
    from .evaluation import evaluation_path, build_evaluation_report
    if evaluation_path(csv_path).exists():
        build_evaluation_report(csv_path, md_path, tag=tag)
        return
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    metadata_path = csv_path.with_name(csv_path.stem + "_run.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    lines = render_report_lines(csv_path, rows, tag=tag, metadata=metadata)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {md_path}")


def render_report_lines(csv_path: Path, rows: list[dict], *, tag: str = "",
                        metadata: dict | None = None, local: bool = False) -> list[str]:
    metadata = metadata if metadata is not None else {}
    by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_key[(r["dataset"], r["service"], r.get("variant_id", ""))].append(r)

    locales_in_run = sorted({r.get("locale") or r["dataset"].rsplit("_", 1)[0] for r in rows})
    locale = locales_in_run[0] if locales_in_run else "en-US"

    ping_ms = None if local else tcp_ping_ms(_HOST)
    ping_str = f"{ping_ms:.1f} ms" if ping_ms is not None else "n/a"
    llm_ping_line = ""
    if not local and _LLM_HOST and _LLM_HOST != _HOST:
        llm_ping = tcp_ping_ms(_LLM_HOST)
        llm_ping_str = f"{llm_ping:.1f} ms" if llm_ping is not None else "n/a"
        llm_ping_line = (f"LLM endpoint host: **{_LLM_HOST}**  \n"
                         f"TCP ping to `{_LLM_HOST}:443` (avg of 5): **{llm_ping_str}**")

    info = {} if local else public_ip_and_location()
    loc_parts = [p for p in (info.get("city"), info.get("region"), info.get("country")) if p]
    loc_str = ", ".join(loc_parts) if loc_parts else "unknown"
    ip_str = info.get("ip") or "unknown"
    org_str = info.get("org") or ""

    title = tag if tag else csv_path.stem
    lines = [f"# ASR Benchmark — {title}", ""]
    lines.append("Report template: **classic-v1**. Generated by fixed Python rendering code.")
    if metadata.get("evaluation"):
        lines.extend(["Language-mode provenance: recorded run options and per-variant settings are in the evaluation appendix.", ""])
    else:
        lines.extend(["Language-mode provenance: **legacy/unknown**. Do not infer fixed-language versus "
                      "automatic behavior from the dataset locale or current adapter defaults.", ""])
    if "pace" in metadata:
        lines.append(f"Real-time pacing: **{metadata['pace']}**. "
                     + ("Latency comparisons require trustworthy shared speech anchors."
                        if metadata["pace"] else
                        "**Unpaced run: do not use these numbers for latency comparisons.**"))
    else:
        lines.append("Pacing provenance is unavailable (legacy artifact); verify the run command before latency comparisons.")
    lines.append(f"Total rows: **{len(rows)}**  ")
    if local:
        lines.append("Network context: **not probed during local report regeneration**. Historical tester IP/location and ping are unavailable.")
        lines.append(f"Azure region: **{metadata.get('speech_region', 'unavailable')}**  ")
        lines.append(f"Azure endpoint: **{metadata.get('speech_endpoint', 'unavailable')}**  ")
        lines.append("VAD/truncation: consult recorded `vad_truncated_s` and run options; current defaults are not historical evidence.")
    else:
        lines.append(f"Tester public IP: **{ip_str}**  ")
        lines.append(f"Tester location: **{loc_str}**" + (f" ({org_str})" if org_str else "") + "  ")
        lines.append(f"Azure region: **{AZURE_SPEECH_REGION}**  ")
        lines.append(f"Azure endpoint host: **{_HOST}**  ")
        lines.append(f"TCP ping to `{_HOST}:443` (avg of 5): **{ping_str}**  ")
        lines.append(f"VAD set to **{SEGMENTATION_SILENCE_MS} ms** (realtime `Speech_SegmentationSilenceTimeoutMs`; fast_* audio truncated at `speech_end + {SEGMENTATION_SILENCE_MS} ms`)")
    if llm_ping_line:
        lines.append("")
        lines.append(llm_ping_line)
    lines.append("")

    # Endpoints under test
    endpoints = _endpoints_for_locale(locale)
    lines.append("## Endpoints under test")
    lines.append("")
    services_in_run = [s for s in endpoints
                       if any(r["service"] == s for r in rows)]
    for svc in services_in_run:
        e = endpoints[svc]
        lines.append(f"### `{svc}` — {e['name']}")
        lines.append(f"- URL: `{metadata.get('speech_endpoint', 'unavailable') if local else e['url']}`")
        lines.append(f"- Transport: {e['transport']}")
        if local:
            lines.append("- Config: recorded mode, locale and variant in the evaluation appendix; current adapter defaults are not run provenance.")
        else:
            lines.append(f"- Config: `{e['config']}`")
        lines.append(f"- Partial results: {e['partials']}")
        if e.get("doc"):
            lines.append(f"- Docs: <{e['doc']}>")
        if not local and e.get("note"):
            lines.append(f"- Note: {e['note']}")
        lines.append("")

    for svc in sorted({r["service"] for r in rows} - endpoints.keys()):
        lines.extend([f"### `{svc}` — historical/unregistered service",
                      "Results are retained; no current endpoint configuration is advertised.", ""])

    # Datasets under test
    datasets_in_run = sorted({r["dataset"] for r in rows})
    lines.append("## Datasets under test")
    lines.append("")
    for ds in datasets_in_run:
        n_samples = len({r["sample_id"] for r in rows if r["dataset"] == ds})
        parts = ds.rsplit("_", 1)
        loc_name = parts[0] if len(parts) == 2 else ds
        scn = parts[1] if len(parts) == 2 else ""
        lines.append(f"- **{ds}** — {loc_name} {scn} ({n_samples} samples)")
    lines.append("")

    # Results table
    lines.append("## Results")
    lines.append("")
    lines.append("WER breakdown columns are *rates per 100 reference words*. "
                 "Per-row WER ≈ (INS + DEL + SUB) / ref_len (capped at 1.0 per sample in aggregation).")
    lines.append("First Latency is speech-anchored streaming first partial, not REST request-to-response time. "
                 "REST First is not applicable; other missing metrics are unavailable, not zero. "
                 "UPL values with different anchors are not directly comparable.")
    if local:
        lines.append("Service labels retain variant IDs; configurations are never pooled. "
                     "Reference-free accuracy is not applicable. For Chinese, WER is whitespace diagnostic only; "
                     "primary CER, character-normalized SER and detailed timing counts/medians are in the evaluation appendix.")
    lines.append("")
    lines.append("| Dataset | Service | N | Errors | WER | SER | INS/100 | DEL/100 | SUB/100 | First Latency ms (mean / p90) | LBL ms (mean / p90) | UPL ms (mean / p90) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")

    from .evaluation_report import _text

    for (ds, svc, variant) in sorted(by_key.keys()):
        rs = by_key[(ds, svc, variant)]
        n = len(rs)
        errs = sum(1 for r in rs if r["error"])
        ok = [r for r in rs if not r["error"]]
        scored = [r for r in ok if r.get("reference_available", True)]
        wers = [min(float(r["wer"]), 1.0) for r in scored if r["wer"] != ""]
        sers = [float(r["sentence_error"]) for r in scored if r["sentence_error"] != ""]
        ll = [float(r["lbl_ms"]) for r in ok if r["lbl_ms"] not in ("", None)]

        ins_total = sum(int(r["ins"]) for r in scored if r.get("ins") not in ("", None))
        del_total = sum(int(r["del"]) for r in scored if r.get("del") not in ("", None))
        sub_total = sum(int(r["sub"]) for r in scored if r.get("sub") not in ("", None))
        ref_total = sum(int(r["ref_len"]) for r in scored if r.get("ref_len") not in ("", None))

        if svc in STREAMING_SERVICES:
            fl = [float(r["first_latency_ms"]) for r in ok if r["first_latency_ms"] not in ("", None)]
        else:
            fl = []
        upl = [float(r["upl_ms"]) for r in ok if r.get("upl_ms") not in ("", None)]

        wer_avg = f"{sum(wers)/len(wers):.3f}" if wers else "-"
        ser_avg = f"{sum(sers)/len(sers):.3f}" if sers else "-"
        ins_rate = f"{100.0 * ins_total / ref_total:.1f}" if ref_total else "-"
        del_rate = f"{100.0 * del_total / ref_total:.1f}" if ref_total else "-"
        sub_rate = f"{100.0 * sub_total / ref_total:.1f}" if ref_total else "-"
        fl_a = aggregate(fl)
        ll_a = aggregate(ll)
        upl_a = aggregate(upl)

        def fmt(a):
            if a["n"] == 0:
                return "-"
            return f"{a['mean']:.0f} / {a['p90']:.0f}"

        label = f"{svc} [{variant}]" if variant else svc
        lines.append(f"| {_text(ds)} | {_text(label)} | {n} | {errs} | {wer_avg} | {ser_avg} | {ins_rate} | {del_rate} | {sub_rate} | {fmt(fl_a)} | {fmt(ll_a)} | {fmt(upl_a)} |")

    lines.append("")

    # Speech boundaries
    lines.append("## Speech boundaries")
    lines.append("")
    lines.append("`speech_start_s` / `speech_end_s` are shared baseline realtime boundaries where available. "
                 "Check `upl_anchor`: self-derived values are not shared-anchor comparisons. "
                 "The sidecar `" + csv_path.with_name(csv_path.stem + "_words.jsonl").name +
                 "` contains service-tagged streaming evidence and available words. Legacy entries without a service are baseline realtime.")
    lines.append("")
    realtime_rows = [r for r in rows if r["service"] == "realtime"]
    fixed_rows = [r for r in realtime_rows
                  if r.get("boundary_fix_action") and r["boundary_fix_action"] != "none"]
    if fixed_rows:
        action_counts = Counter(r["boundary_fix_action"] for r in fixed_rows)
        lines.append(f"Boundary-fix decisions across {len(realtime_rows)} realtime samples:")
        lines.append("")
        for action, n in sorted(action_counts.items()):
            lines.append(f"- `{action}`: {n}")
        lines.append("")
        lines.append("Trimmed/skipped samples (first 20):")
        lines.append("")
        lines.append("| Dataset | Sample ID | Action | speech_start_s | speech_end_s |")
        lines.append("|---|---|---|---:|---:|")
        for r in fixed_rows[:20]:
            ss = r.get("speech_start_s", "") or "-"
            se = r.get("speech_end_s", "") or "-"
            lines.append(f"| {_text(r['dataset'])} | {_text(r['sample_id'])} | {_text(r['boundary_fix_action'])} | {ss} | {se} |")
        lines.append("")
    elif realtime_rows:
        lines.append("All realtime samples passed the boundary-fix heuristic without trimming.")
        lines.append("")
    else:
        lines.append("No baseline realtime boundary checks were available.")

    # Latency definitions
    lines.append("## Latency definitions (all values in ms)")
    lines.append("")
    lines.append("Wall-clock timeline for one realtime sample. The clip has leading and")
    lines.append("trailing silence; the user only speaks during the middle. Word offsets")
    lines.append("come from the SDK's word-level timestamps.")
    lines.append("")
    lines.append("```")
    lines.append("  audio_start         speech_start                  speech_end       end_of_audio")
    lines.append("       │                   │                              │                │")
    lines.append("       ├───────────────────┼──────────────────────────────┼────────────────┤")
    lines.append("       │  leading silence  │        user speaking         │trailing silence│")
    lines.append("       │                   │                              │                │")
    lines.append("       │                   │   first_recognizing event    │                │   last_recognized event")
    lines.append("       │                   │             ▼                │                │           ▼")
    lines.append("       │                   ├────────────→│ First Latency  │                │           │")
    lines.append("       │                   │                              │                │           │")
    lines.append("       │                   │                              ├────────────────┼──────────→│ UPL")
    lines.append("       │                   │                              │                │           │")
    lines.append("       │                   │                              │                ├──────────→│ LBL  (can be negative)")
    lines.append("```")
    lines.append("")
    lines.append("Reference points on the timeline:")
    lines.append("- `audio_start` — wall time of the first PCM chunk we push.")
    lines.append("- `speech_start` = `audio_start + first_word_start_in_audio` — first word's `Offset` from the first `recognized` event.")
    lines.append("- `speech_end` = `audio_start + last_word_end_in_audio` — last word's `Offset + Duration` from the last `recognized` event.")
    lines.append("- `end_of_audio` — wall time when the last PCM chunk has been pushed.")
    lines.append("")
    lines.append("**realtime** (Speech SDK, partials available):")
    lines.append("- **First Latency** = `first_recognizing_event_wall − speech_start`. Subtracts leading silence so the metric reflects how fast the first partial appears *after the user starts speaking*, not after we start streaming bytes.")
    lines.append("- **LBL** (last-final beyond last-chunk) = `last_recognized_event_wall − end_of_audio`. Observed final-event delay relative to upload completion, not isolated server processing time. **May be negative** when the SDK emits the final result before the last chunk goes out.")
    lines.append("- **UPL** (user-perceived latency) = `last_recognized_event_wall − speech_end`. How late the final result arrives after the user actually stopped speaking.")
    lines.append("")
    lines.append("**fast_default / fast_llm / fast_mai_1_5 / fast_mai_2** (REST, no partials):")
    lines.append("- **LBL** = `response_fully_read_wall − end_of_audio_wall`. Time from last uploaded byte to fully-received response.")
    lines.append("- **UPL** = `response_fully_read_wall − speech_end_wall`. `speech_end_wall` is taken from the realtime SDK's last-word offset for that sample (when realtime ran successfully on it), so all services compare against the same reference. The CSV column `upl_self_ms` keeps each service's own phrase-derived value, and `upl_anchor` is `realtime` when anchored or `self` when the realtime anchor was unavailable.")
    lines.append("- Note: MAI phrase boundaries can span the full audio; using the realtime anchor here makes UPL directly comparable across services.")
    lines.append("- **First Latency** is omitted because the REST response is delivered in one shot — there is no first-token signal to measure against.")
    lines.append(f"- **VAD truncation**: when realtime ran first and produced word timestamps, fast_* audio is truncated at `speech_end + {SEGMENTATION_SILENCE_MS} ms` to simulate a VAD cutting the stream after end-of-speech. The CSV column `vad_truncated_s` records the truncated duration.")
    lines.append("")
    lines.append("**Other:**")
    lines.append("- **realtime_mai_2** uses the same first-partial/LBL definitions. First latency and UPL use baseline boundaries when available; otherwise they require returned model word timestamps. Missing metrics are unavailable, not zero. `upl_self_ms` is retained separately.")
    lines.append("- **realtime_refine** uses the SDK event definitions above but may use its own word boundaries; a legacy `realtime` anchor label alone does not establish a shared baseline.")
    lines.append("- **realtime_mai_2_api** uses first intermediate and final-completion events; explicit audio commit differs from SDK EOF. Call-to-first-text includes setup and is not speech-anchored First Latency.")
    lines.append("- **azure_whisper** is non-streaming: First is omitted from the headline table. Its recorded request-to-final-response timing is not streaming First; LBL starts at upload completion and UPL requires a speech-end anchor.")
    lines.append("- Timing counts/medians and anchor distributions, when recorded, remain in the evaluation appendix. Do not rank unpaced runs or incompatible anchors. Negative First can reflect partial/final word alignment, not negative elapsed request time.")
    lines.append("- WER/SER use NFKC + lowercase + punctuation stripping normalization.")
    lines.append("- Per-sample WER is capped at 1.0 in aggregation.")

    return lines
