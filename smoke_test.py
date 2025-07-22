#!/usr/bin/env python
import os, sys, asyncio
from pathlib import Path
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).parent
SERVER = ROOT / "gitlab_mcp.py"

load_dotenv()

PROMPT = "Explain how AI works in a few words"
PROJECT_LIMIT = 3

async def run():
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        working_directory=str(ROOT),
    )

    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            print("[1] Init MCP…", flush=True)
            await s.initialize()

            print("[2] Test LLM…", flush=True)
            llm = await s.call_tool("gemini_generate", {"prompt": PROMPT, "model": "gemini-2.0-flash"})
            text = (llm.structuredContent or {}).get("result")
            if not text and getattr(llm, "content", None):
                text = getattr(llm.content[0], "text", None)
            assert text, "LLM lieferte keinen Text"
            print("    ✓", text[:120].replace("\n", " ") + ("…" if len(text) > 120 else ""))

            print("[3] Test GitLab…", flush=True)
            res = await s.call_tool("list_projects", {"limit": PROJECT_LIMIT})
            data = (res.structuredContent or {}).get("result")
            if data is None and getattr(res, "content", None):
                data = getattr(res.content[0], "data", None)
            assert isinstance(data, list) and data, "Keine Projekte gefunden"
            print(f"    ✓ {len(data)} Projekte (zeige {PROJECT_LIMIT}):")
            for p in data[:PROJECT_LIMIT]:
                print("      -", p.get("name", p))

    print("\nAlles OK ✅")

def main():
    try:
        asyncio.run(run())
    except Exception as e:
        print("Smoke-Test FAILED ❌:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
