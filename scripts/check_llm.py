"""Check that the configured model API works (one tiny request per model role).

    python scripts/check_llm.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.ai.provider_factory import ChatRequest, ProviderError, ProviderFactory  # noqa: E402
from backend.config import get_settings  # noqa: E402


def main() -> int:
    s = get_settings()
    key = s.openai_api_key or s.anthropic_api_key
    print(f"mode={s.provider_mode}")
    print(f"openai_base_url={s.openai_base_url}")
    print(f"api key set: {'yes' if key else 'NO'} (length {len(key)})")
    if s.is_mock:
        print("LLM_PROVIDER_MODE is 'mock' -> no API calls are made. Set LLM_PROVIDER_MODE=live in .env")
        return 1
    factory = ProviderFactory(s)
    ok = True
    for role in ("primary", "synthesis", "light", "fallback"):
        ref = getattr(s, f"model_{role}")
        t0 = time.perf_counter()
        try:
            resp = factory.get(ref).complete(ChatRequest(
                task="check", system='Reply with exactly this JSON and nothing else: {"ok": true}',
                messages=[{"role": "user", "content": "ping"}], max_tokens=50))
            print(f"OK    {role:<9} {ref:<40} {time.perf_counter() - t0:5.1f}s  reply: {resp.text.strip()[:60]!r}")
        except ProviderError as e:
            ok = False
            print(f"FAIL  {role:<9} {ref:<40} kind={e.kind} status={e.status}\n      {str(e)[:400]}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
