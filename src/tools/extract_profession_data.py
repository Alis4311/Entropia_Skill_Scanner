from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests


LIST_URL   = "https://api.entropiacentral.com/api/skills/professions/"
DETAIL_URL = "https://api.entropiacentral.com/api/skills/professions/{name}"

OUT_DIR   = Path("out_ec_professions")
DETAILS   = OUT_DIR / "details"
META_DIR  = OUT_DIR / "http_meta"
OUT_JSON  = OUT_DIR / "professions.json"

OUT_DIR.mkdir(exist_ok=True)
DETAILS.mkdir(exist_ok=True)
META_DIR.mkdir(exist_ok=True)

# ---- politeness knobs ----
REQUESTS_PER_SECOND = 1.0     # drop to 0.5 if you want extra gentle
TIMEOUT_S = 30
MAX_RETRIES = 8
BASE_BACKOFF = 1.0
MAX_BACKOFF = 60.0
JITTER = 0.5


def strip_category(name: str) -> str:
    return name.split(":", 1)[1].strip() if ":" in name else name.strip()


@dataclass
class RateLimiter:
    min_interval: float
    next_ok: float = 0.0

    def wait(self) -> None:
        now = time.time()
        if now < self.next_ok:
            time.sleep(self.next_ok - now)
        self.next_ok = time.time() + self.min_interval


def meta_paths(key: str) -> tuple[Path, Path]:
    safe = key.replace("/", "_")
    return META_DIR / f"{safe}.etag", META_DIR / f"{safe}.lm"


def polite_get(
    s: requests.Session,
    url: str,
    *,
    cache_key: Optional[str],
    limiter: RateLimiter,
) -> Optional[Any]:
    headers: Dict[str, str] = {}

    if cache_key:
        etag_p, lm_p = meta_paths(cache_key)
        if etag_p.exists():
            headers["If-None-Match"] = etag_p.read_text()
        if lm_p.exists():
            headers["If-Modified-Since"] = lm_p.read_text()

    for attempt in range(MAX_RETRIES):
        limiter.wait()
        try:
            r = s.get(url, headers=headers, timeout=TIMEOUT_S)

            if r.status_code == 304:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(str(r.status_code), response=r)

            r.raise_for_status()

            if cache_key:
                etag, lm = meta_paths(cache_key)
                if r.headers.get("ETag"):
                    etag.write_text(r.headers["ETag"])
                if r.headers.get("Last-Modified"):
                    lm.write_text(r.headers["Last-Modified"])

            return r.json()

        except Exception as e:
            delay = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** attempt))
            delay += random.random() * JITTER
            time.sleep(delay)

    raise RuntimeError(f"Failed after retries: {url}")


def main() -> int:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "EntropiaProfessionExtractor/1.0 (polite; 1rps)",
        "Accept": "application/json",
    })

    limiter = RateLimiter(1.0 / REQUESTS_PER_SECOND)

    prof_list = polite_get(s, LIST_URL, cache_key="list", limiter=limiter)
    if not isinstance(prof_list, list):
        raise RuntimeError("Unexpected professions list payload")

    out: Dict[str, list[dict[str, int]]] = {}

    for i, p in enumerate(prof_list, 1):
        name = p["activityName"]
        safe_name = quote(name, safe="")
        detail_path = DETAILS / f"{safe_name}.json"

        if detail_path.exists():
            detail = json.loads(detail_path.read_text(encoding="utf-8"))
        else:
            detail = polite_get(
                s,
                DETAIL_URL.format(name=safe_name),
                cache_key=f"detail_{safe_name}",
                limiter=limiter,
            )
            if detail is None:
                raise RuntimeError(f"304 but no cache for {name}")
            detail_path.write_text(json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")

        skills = [
            {
                "skill": strip_category(sk["skillName"]),
                "pct": int(sk["skillEffect"]),
            }
            for sk in detail.get("professionSkills", [])
            #if not sk.get("hidden")
        ]

        out[name] = skills

        if i % 25 == 0:
            print(f"Processed {i}/{len(prof_list)}")

    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
