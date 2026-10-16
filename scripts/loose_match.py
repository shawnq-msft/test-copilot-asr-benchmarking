"""Apply automatic loose-match normalization and recompute WER on a benchmark CSV.

Handles systematic false-positive WER patterns caused by formatting differences,
not recognition errors:
  - Compound-word splitting:  stummschalten ↔ stumm schalten
  - Alphanumeric spacing:     2D ↔ 2 D
  - Percent spacing:          60% ↔ 60 %
  - Degree symbol variants:   360° ↔ 360 gradi / degrees / grad / grados / degrés

Strategy: after standard normalize_text, if ref and hyp differ only by
word boundaries / known symbol equivalences, set WER=0. Otherwise keep
the standard WER.

Backs up the strict CSV to <run>.csv.strict, then overwrites
wer/ins/del/sub/ref_len/sentence_error with loose-match values.
Chinese character-based SER and all CER fields retain their strict policy.

After running, re-run report.py and error_analysis.py to refresh markdown.

Usage:
    python -X utf8 scripts/loose_match.py results/<run>.csv
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.metrics import normalize_text  # noqa: E402

_DEGREE_WORDS = {
    "gradi", "degrees", "degree", "grad", "grados", "grado",
    "degrés", "degré", "degres",
}


def _canonicalize(s: str) -> str:
    """Reduce a normalized string to a canonical form for loose comparison.

    After normalize_text (NFKC + lowercase + punct-strip + space-collapse):
      1. Replace a number followed by a degree word with canonical token
      2. Remove all spaces (catches compound splits, digit-letter spacing)
    """
    words = s.split()
    canonical: list[str] = []
    i = 0
    while i < len(words):
        if i + 1 < len(words) and re.match(r"\d+$", words[i]) and words[i + 1] in _DEGREE_WORDS:
            canonical.append(words[i])
            canonical.append("DEG")
            i += 2
        else:
            canonical.append(words[i])
            i += 1
    return "".join(canonical)


def _is_loose_match(ref: str, hyp: str) -> bool:
    """Return True if ref and hyp are equivalent under loose matching."""
    nr = normalize_text(ref)
    nh = normalize_text(hyp)
    if nr == nh:
        return True
    return _canonicalize(nr) == _canonicalize(nh)


def main(csv_path: Path) -> None:
    from benchmark.evaluation import evaluation_path
    if evaluation_path(csv_path).exists():
        entries = [json.loads(line) for line in evaluation_path(csv_path).read_text(encoding="utf-8").splitlines()]
        if any(e["reference_segments"] or e["options"]["diarization"]
               or e["options"]["language_mode"] == "multilingual" for e in entries):
            raise ValueError("Annotated evaluation uses strict token ownership; loose matching is disabled. "
                             "Preserve the original evaluation artifacts.")
    backup = csv_path.with_suffix(csv_path.suffix + ".strict")
    if not backup.exists():
        shutil.copy2(csv_path, backup)
        print(f"[loose_match] backed up strict CSV -> {backup.name}")

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for r in reader:
            rows.append(r)

    n_changed = 0
    for r in rows:
        if r.get("error") or not r.get("reference"):
            continue
        ref = r["reference"]
        hyp = r.get("hypothesis", "") or ""

        old_wer = r.get("wer", "")
        if not old_wer or float(old_wer) == 0.0:
            continue

        if _is_loose_match(ref, hyp):
            ref_words = normalize_text(ref).split()
            r["wer"] = 0.0
            r["ins"] = 0
            r["del"] = 0
            r["sub"] = 0
            r["ref_len"] = len(ref_words)
            if not (r.get("locale", "").startswith(("zh-", "yue-")) and r.get("cer") not in ("", None)):
                r["sentence_error"] = 0
            n_changed += 1

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[loose_match] zeroed WER on {n_changed} row(s) via loose normalization")
    print("[loose_match] re-run report.py and error_analysis.py to refresh markdown outputs.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -X utf8 scripts/loose_match.py <csv>")
        sys.exit(2)
    main(Path(sys.argv[1]))
