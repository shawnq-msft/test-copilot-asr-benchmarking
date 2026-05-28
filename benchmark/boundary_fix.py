from __future__ import annotations

from .metrics import normalize_text, wer_breakdown

WER_GATE = 0.5
INS_RATE_GATE = 0.3


def decide(reference: str, hypothesis: str,
           words: list[dict] | None) -> tuple[float | None, float | None, dict]:
    if not words:
        return None, None, {"action": "skip", "reason": "no words"}

    bd = wer_breakdown(reference, hypothesis)
    ref_len = max(bd["ref_len"], 1)
    wer_val = (bd["ins"] + bd["dele"] + bd["sub"]) / ref_len
    ins_rate = bd["ins"] / ref_len

    fs = words[0]["start_s"]
    le = words[-1]["end_s"]

    if wer_val <= WER_GATE and ins_rate <= INS_RATE_GATE:
        return fs, le, {"action": "none", "reason": ""}

    ref_n = normalize_text(reference)
    first_n = normalize_text(words[0]["text"]) if words else ""
    last_n = normalize_text(words[-1]["text"]) if len(words) > 1 else ""

    drop_first = bool(first_n) and first_n not in ref_n
    drop_last = bool(last_n) and last_n not in ref_n

    if drop_first and drop_last and len(words) >= 3:
        return (words[1]["start_s"], words[-2]["end_s"],
                {"action": "trim_both",
                 "reason": f"edges '{words[0]['text']}'/'{words[-1]['text']}' not in ref; wer={wer_val:.2f}"})
    if drop_first and len(words) >= 2:
        return (words[1]["start_s"], le,
                {"action": "trim_first",
                 "reason": f"'{words[0]['text']}' not in ref; ins_rate={ins_rate:.2f}"})
    if drop_last and len(words) >= 2:
        return (fs, words[-2]["end_s"],
                {"action": "trim_last",
                 "reason": f"'{words[-1]['text']}' not in ref; ins_rate={ins_rate:.2f}"})

    return None, None, {"action": "skip",
                        "reason": f"high wer={wer_val:.2f} but edges look aligned; anchor unreliable"}
