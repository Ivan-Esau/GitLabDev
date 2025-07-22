import json
from typing import Any, Dict, List, Optional, Set

import httpx
from fastapi import FastAPI

from .config import (
    LLM_PROVIDER,
    DEFAULT_MODEL,
    GEMINI_API_KEY,
    GEMINI_TOOL_MODE,
    GEMINI_URL_TMPL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    SYSTEM_PROMPT,
)
from .mcp_client import call_mcp_tool, update_context, autofill_args, ensure_project_index


# ---------------- LLM CALLS ----------------
async def llm_call(messages: List[Dict[str, Any]],
                   tools_block: List[Dict[str, Any]],
                   model: Optional[str]) -> Dict[str, Any]:
    """
    Wrapper für verschiedene Provider. Aktuell:
      - gemini (voll unterstützt inkl. tools)
      - openai (rudimentär: Text, kein automatisches Tool-Handling hier integriert)
    """
    provider = LLM_PROVIDER

    if provider == "gemini":
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": messages,
            "tools": tools_block,
            "toolConfig": {"functionCallingConfig": {"mode": GEMINI_TOOL_MODE}},
        }
        url = GEMINI_URL_TMPL.format(model=model or DEFAULT_MODEL)
        headers = {"Content-Type": "application/json", "X-Goog-Api-Key": GEMINI_API_KEY}
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()

    elif provider == "openai":
        # NOTE: Hier nur ein Minimal-Pfad ohne Tool-Calls. Für echte Tools müsste das Message/Tool-Mapping angepasst werden.
        oai_msgs = _gemini_to_openai(messages)
        payload = {
            "model": model or DEFAULT_MODEL,
            "messages": oai_msgs,
            "temperature": 0.2,
        }
        url = f"{OPENAI_BASE_URL}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return _openai_to_gemini_like(resp.json())

    else:
        raise RuntimeError(f"LLM_PROVIDER '{provider}' wird noch nicht unterstützt.")


def _gemini_to_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Sehr einfache Konvertierung: nimmt nur Text-Parts."""
    out = []
    for m in messages:
        role = "user" if m["role"] == "user" else "assistant"
        text_parts = []
        for p in m.get("parts", []):
            if "text" in p:
                text_parts.append(p["text"])
        if not text_parts:
            continue
        out.append({"role": role, "content": "\n".join(text_parts)})
    return out


def _openai_to_gemini_like(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Normalisiert OpenAI-Antwort auf 'gemini-like' Struktur (nur Text)."""
    choice = resp.get("choices", [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content", "")
    return {
        "candidates": [{
            "content": {
                "parts": [{"text": text}]
            }
        }]
    }


# ---------------- EXTRACTORS ----------------
def extract_function_calls(resp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    # Gemini-style
    for cand in resp_json.get("candidates", []):
        for p in cand.get("content", {}).get("parts", []):
            if "functionCall" in p:
                calls.append(p["functionCall"])
    return calls

def extract_text(resp_json: Dict[str, Any]) -> Optional[str]:
    for cand in resp_json.get("candidates", []):
        for p in cand.get("content", {}).get("parts", []):
            if "text" in p:
                return p["text"]
    return None


# ---------------- AGENT LOOP ----------------
async def run_agent(app: FastAPI, user_msg: str, max_steps: int, model: Optional[str], sid: str) -> str:
    conv = app.state.conversations.setdefault(sid, [])
    ctx = app.state.contexts.setdefault(sid, {})

    # Projektindex injizieren, falls kein Projekt im Kontext
    if not ctx.get("project_ref"):
        await ensure_project_index(app, sid, conv)

    # Nutzer-Message anhängen
    conv.append({"role": "user", "parts": [{"text": user_msg}]})

    seen_calls: Set[str] = set()
    tool_only_turns = 0
    TOOL_TURN_LIMIT = 3
    SKIP_WHEN_PROJECT_KNOWN = {"list_projects", "resolve_project"}

    for _ in range(max_steps):
        resp = await llm_call(conv, app.state.gemini_tool_block, model)
        parts = resp["candidates"][0]["content"]["parts"]
        conv.append({"role": "model", "parts": parts})

        calls = extract_function_calls(resp)
        if not calls:
            return extract_text(resp) or json.dumps(resp)

        response_parts = []
        for call in calls:
            name = call["name"]

            # unnötige Tools überspringen
            if ctx.get("project_ref") and name in SKIP_WHEN_PROJECT_KNOWN:
                continue

            sig = f"{name}:{json.dumps(call.get('args', {}), sort_keys=True)}"
            if sig in seen_calls:
                continue
            seen_calls.add(sig)

            args = autofill_args(app, sid, name, call.get("args", {}) or {})
            try:
                result = await call_mcp_tool(app, name, args)
            except Exception as e:
                result = {"error": str(e)}
            update_context(app, sid, name, result)

            response_parts.append({
                "functionResponse": {
                    "name": name,
                    "response": {"result": result}
                }
            })

        if not response_parts:
            # keine neuen Antworten -> Abschluss erzwingen
            conv.append({"role": "user", "parts": [{"text": "Beantworte jetzt final, ohne Tools."}]})
            continue

        conv.append({"role": "user", "parts": response_parts})
        tool_only_turns += 1
        if tool_only_turns >= TOOL_TURN_LIMIT:
            conv.append({"role": "user", "parts": [{"text": "Fasse alles zusammen und antworte endgültig."}]})
            tool_only_turns = 0

    return "[Agent aborted: max_steps reached]"
