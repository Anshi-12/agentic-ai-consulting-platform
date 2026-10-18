"""Download the web pages and datasheets the professor listed as links, so the
Technology and Clinical agents can cite their full text.

    python scripts/fetch_web_sources.py          # then: python -m backend.rag.ingest

Sources: data/hackathon/DataSource-Cisco.txt, Agilent.txt, Google GCP.txt, plus
optional data/hackathon/extra_sources.txt with lines "domain | vendor | name | url"
(use it to add IBM documentation, which the export does not include).
Saves to data/hackathon/web/ and writes web/manifest.json. Run it on your own
machine; sites that block automated downloads are recorded as failed.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from backend.rag.ingest import DEFAULT_DATA, catalog_entries  # noqa: E402

WEB = DEFAULT_DATA / "web"
UA = "Mozilla/5.0 (X11; Linux x86_64) SJSU-CMPE256-hackathon-research-bot/1.0"


class TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in {"p", "li", "h1", "h2", "h3", "h4", "tr", "br", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip() + " ")


def html_to_text(html: str) -> str:
    p = TextExtractor()
    p.feed(html)
    text = re.sub(r"[ \t]+", " ", "".join(p.parts))
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def extra_entries() -> list[dict]:
    f = DEFAULT_DATA / "extra_sources.txt"
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 4 and parts[3].startswith("http") and parts[0] in ("technology", "clinical"):
            out.append({"domain": parts[0], "vendor": parts[1], "name": parts[2], "url": parts[3]})
    return out


def main() -> int:
    if not DEFAULT_DATA.exists():
        print("Run scripts/setup_hackathon_data.py first.")
        return 1
    WEB.mkdir(parents=True, exist_ok=True)
    entries = catalog_entries(DEFAULT_DATA) + extra_entries()
    manifest = []
    with httpx.Client(follow_redirects=True, timeout=30, headers={"user-agent": UA}) as client:
        for e in entries:
            key = hashlib.sha1(e["url"].encode()).hexdigest()[:12]
            rec = {**{k: e[k] for k in ("domain", "vendor", "name", "url")}, "status": "failed"}
            try:
                r = client.get(e["url"])
                r.raise_for_status()
                ctype = r.headers.get("content-type", "")
                if "pdf" in ctype or e["url"].lower().endswith(".pdf"):
                    path = WEB / e["domain"] / f"{key}.pdf"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(r.content)
                else:
                    text = html_to_text(r.text)
                    if len(text) < 200:
                        raise ValueError("page has too little text (likely rendered by JavaScript)")
                    path = WEB / e["domain"] / f"{key}.txt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text, encoding="utf-8")
                rec.update(status="ok", path=str(path.relative_to(DEFAULT_DATA)), bytes=path.stat().st_size)
                print(f"ok      {e['vendor']:8} {e['name'][:60]}")
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                print(f"failed  {e['vendor']:8} {e['name'][:60]}  ({rec['error']})")
            manifest.append(rec)
            time.sleep(0.5)  # be polite
    (WEB / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok = sum(m["status"] == "ok" for m in manifest)
    print(f"\n{ok}/{len(manifest)} fetched. Rebuild the knowledge base: python -m backend.rag.ingest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
