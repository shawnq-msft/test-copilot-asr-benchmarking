import re
import unicodedata

from rapidfuzz.distance import Levenshtein

_PUNCT_RE = re.compile(r"[^\w\s]|_", re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s


def wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return Levenshtein.distance(ref_words, hyp_words) / len(ref_words)


def wer_breakdown(reference: str, hypothesis: str) -> dict:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return {"ins": len(hyp_words), "dele": 0, "sub": 0, "ref_len": 0}
    ops = Levenshtein.editops(ref_words, hyp_words)
    ins = sum(1 for op in ops if op.tag == "insert")
    dele = sum(1 for op in ops if op.tag == "delete")
    sub = sum(1 for op in ops if op.tag == "replace")
    return {"ins": ins, "dele": dele, "sub": sub, "ref_len": len(ref_words)}


def sentence_error(reference: str, hypothesis: str) -> int:
    return 0 if normalize_text(reference) == normalize_text(hypothesis) else 1


def aggregate(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    vs = sorted(values)
    n = len(vs)
    return {
        "n": n,
        "mean": sum(vs) / n,
        "median": vs[n // 2] if n % 2 else (vs[n // 2 - 1] + vs[n // 2]) / 2,
        "p90": vs[min(int(n * 0.9), n - 1)],
        "min": vs[0],
        "max": vs[-1],
    }
