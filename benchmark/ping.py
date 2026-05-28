from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import requests

_CACHE_PATH = Path(__file__).resolve().parent.parent / "results" / ".tester_info.json"


def tcp_ping_ms(host: str, port: int = 443, count: int = 5,
                timeout: float = 3.0) -> float | None:
    samples: list[float] = []
    for _ in range(count):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            t0 = time.perf_counter()
            s.connect((host, port))
            samples.append((time.perf_counter() - t0) * 1000.0)
        except OSError:
            continue
        finally:
            try:
                s.close()
            except OSError:
                pass
    if not samples:
        return None
    return sum(samples) / len(samples)


def public_ip_and_location(timeout: float = 5.0, use_cache: bool = True) -> dict:
    if use_cache:
        cached = _load_cache()
        if cached:
            return cached
    out = {"ip": None, "city": None, "region": None, "country": None, "org": None}
    providers = [
        ("https://ipapi.co/json/", lambda j: {
            "ip": j.get("ip"), "city": j.get("city"), "region": j.get("region"),
            "country": j.get("country_name") or j.get("country"), "org": j.get("org"),
            "_ok": not j.get("error"),
        }),
        ("https://ipinfo.io/json", lambda j: {
            "ip": j.get("ip"), "city": j.get("city"), "region": j.get("region"),
            "country": j.get("country"), "org": j.get("org"),
            "_ok": True,
        }),
    ]
    for url, parser in providers:
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                continue
            parsed = parser(r.json())
            if not parsed.pop("_ok", False):
                continue
            for k, v in parsed.items():
                if v and not out.get(k):
                    out[k] = v
            if _is_complete(out):
                break
        except Exception:
            continue
    if _is_complete(out):
        _save_cache(out)
    return out


def _is_complete(info: dict) -> bool:
    return bool(info.get("ip") and info.get("city") and info.get("country"))


def _load_cache() -> dict | None:
    try:
        if _CACHE_PATH.exists():
            data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            if _is_complete(data):
                return data
    except Exception:
        pass
    return None


def _save_cache(info: dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    except Exception:
        pass
