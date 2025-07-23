# app/routes/local_llm.py
from __future__ import annotations

import json
from typing import List

import httpx
from fastapi import APIRouter, HTTPException, Query
from httpx import ConnectError, HTTPStatusError

from ..config import OLLAMA_BASE_URL

router = APIRouter(tags=["local-llm"])


@router.get("/local_llm/models")
async def list_local_models(base_url: str = OLLAMA_BASE_URL):
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=10) as cli:
            r = await cli.get(url)
            r.raise_for_status()
    except ConnectError:
        raise HTTPException(503, f"Ollama nicht erreichbar unter {base_url}")
    except HTTPStatusError as e:
        raise HTTPException(502, f"Ollama /api/tags Fehler: {e.response.text}")
    return r.json()


@router.post("/local_llm/pull")
async def pull_model(
    model: str = Query(..., description="z.B. deepseek-r1:7b"),
    base_url: str = OLLAMA_BASE_URL,
):
    url = f"{base_url.rstrip('/')}/api/pull"
    payload = {"model": model}
    logs: List[dict] = []

    try:
        async with httpx.AsyncClient(timeout=None) as cli:
            async with cli.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        logs.append({"raw": line})
    except ConnectError:
        raise HTTPException(503, f"Ollama nicht erreichbar unter {base_url}. Läuft 'ollama serve'?")
    except HTTPStatusError as e:
        text = await e.response.aread()
        raise HTTPException(502, f"Ollama /api/pull Fehler: {text.decode(errors='ignore')}")

    status = next((l.get("status") for l in reversed(logs) if isinstance(l, dict)), None)
    return {"model": model, "status": status or "unknown", "log": logs}
