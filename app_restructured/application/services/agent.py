# app/agent.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from fastapi import FastAPI

from app_restructured.core.config import (
    LLM_PROVIDER,
    DEFAULT_MODEL,
    GEMINI_API_KEY,
    GEMINI_TOOL_MODE,
    GEMINI_URL_TMPL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OLLAMA_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
    LLM_TIMEOUT,
    SYSTEM_PROMPT,
)
from app_restructured.infrastructure.mcp.mcp_client import call_mcp_tool, update_context, autofill_args, ensure_project_index


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def strip_think(text: str) -> str:
    """
    Entfernt DeepSeek-R1 <think>...</think>-Blöcke.
    Nimmt den Teil NACH dem letzten </think>, falls vorhanden.
    """
    if "<think" in text:
        # alles hinter dem letzten closing-tag nehmen
        parts = text.rsplit("</think>", 1)
        return parts[-1].strip()
    return text


def build_local_schema(tool_names: List[str]) -> Dict[str, Any]:
    """
    JSON-Schema, das wir an Ollama über 'format' übergeben.
    Es zwingt das Modell, entweder einen Tool-Call oder eine finale Antwort zurückzugeben.
    """
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["tool", "final"]},
            "tool": {"type": "string", "enum": tool_names},
            "arguments": {"type": "object"},
            "final": {"type": "string"},
            "thought": {"type": "string"},
        },
        "required": ["action"],
        "additionalProperties": False,
    }


def parse_local_json(payload: Dict[str, Any]) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
    """
    Erwartet Dict entsprechend dem Schema:
      action: tool|final
      tool: <name>
      arguments: {}
      final: <answer>
    Rückgabe: (action, tool, args, final)
    """
    action = payload.get("action")
    tool = payload.get("tool")
    args = payload.get("arguments", {}) or {}
    final = payload.get("final")
    return action, tool, args, final


# ------------------------------------------------------------
# LLM calls (provider specific)
# ------------------------------------------------------------

async def llm_call_gemini(messages: List[Dict[str, Any]], tools_block: List[Dict[str, Any]], model: Optional[str]) -> Dict[str, Any]:
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": messages,
        "tools": tools_block,
        "toolConfig": {"functionCallingConfig": {"mode": GEMINI_TOOL_MODE}},
    }
    url = GEMINI_URL_TMPL.format(model=model or DEFAULT_MODEL)
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": GEMINI_API_KEY}
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        resp = await client.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


def _gemini_to_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for m in messages:
        role = "user" if m["role"] == "user" else "assistant"
        text_parts = [p["text"] for p in m.get("parts", []) if "text" in p]
        if not text_parts:
            continue
        out.append({"role": role, "content": "\n".join(text_parts)})
    return out


def _openai_to_gemini_like(resp: Dict[str, Any]) -> Dict[str, Any]:
    choice = resp.get("choices", [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content", "")
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


async def llm_call_openai(messages: List[Dict[str, Any]], model: Optional[str]) -> Dict[str, Any]:
    oai_msgs = _gemini_to_openai(messages)
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": oai_msgs,
        "temperature": 0.2,
    }
    url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        resp = await client.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return _openai_to_gemini_like(resp.json())


def _gemini_to_ollama(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for m in messages:
        role = "user" if m["role"] == "user" else "assistant"
        text_parts = [p["text"] for p in m.get("parts", []) if "text" in p]
        if not text_parts:
            continue
        out.append({"role": role, "content": "\n".join(text_parts)})
    return out


def _ollama_to_gemini_like_text(resp: Dict[str, Any]) -> str:
    # /api/chat -> {"message":{"role":"assistant","content":"..."}}
    msg = resp.get("message") or {}
    return msg.get("content", "") if isinstance(msg, dict) else ""


async def llm_call_ollama(
    messages: List[Dict[str, Any]],
    model: Optional[str],
    json_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ollama_msgs = _gemini_to_ollama(messages)
    payload: Dict[str, Any] = {
        "model": model or OLLAMA_DEFAULT_MODEL,
        "messages": ollama_msgs,
        "stream": False,
    }
    if json_schema:
        payload["format"] = json_schema  # Ollama structured output
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    headers = {"Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        resp = await client.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


def extract_function_calls_gemini(resp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for cand in resp_json.get("candidates", []):
        for p in cand.get("content", {}).get("parts", []):
            if "functionCall" in p:
                calls.append(p["functionCall"])
    return calls


def extract_text_gemini(resp_json: Dict[str, Any]) -> Optional[str]:
    for cand in resp_json.get("candidates", []):
        for p in cand.get("content", {}).get("parts", []):
            if "text" in p:
                return p["text"]
    return None


# ------------------------------------------------------------
# Agent Loops
# ------------------------------------------------------------

async def run_agent(
    app: FastAPI,
    user_msg: str,
    max_steps: int,
    model: Optional[str],
    sid: str,
    provider: Optional[str] = None,
) -> str:
    provider = (provider or LLM_PROVIDER).lower()

    if provider == "gemini":
        return await _run_agent_gemini(app, user_msg, max_steps, model, sid)

    # Provider ohne native FunctionCalls (ollama, openai, ...)
    return await _run_agent_local_tools(app, user_msg, max_steps, model, sid, provider)


async def _run_agent_gemini(
    app: FastAPI,
    user_msg: str,
    max_steps: int,
    model: Optional[str],
    sid: str,
) -> str:
    conv = app.state.conversations.setdefault(sid, [])
    ctx = app.state.contexts.setdefault(sid, {})

    if not ctx.get("project_ref"):
        await ensure_project_index(app, sid, conv)

    conv.append({"role": "user", "parts": [{"text": user_msg}]})

    seen_calls: Set[str] = set()
    tool_only_turns = 0
    TOOL_TURN_LIMIT = 3
    SKIP_WHEN_PROJECT_KNOWN = {"list_projects", "resolve_project"}

    for _ in range(max_steps):
        resp = await llm_call_gemini(conv, app.state.gemini_tool_block, model)
        parts = resp["candidates"][0]["content"]["parts"]
        conv.append({"role": "model", "parts": parts})

        calls = extract_function_calls_gemini(resp)
        if not calls:
            return extract_text_gemini(resp) or json.dumps(resp)

        response_parts = []
        for call in calls:
            name = call["name"]
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
                "functionResponse": {"name": name, "response": {"result": result}}
            })

        if not response_parts:
            conv.append({"role": "user", "parts": [{"text": "Beantworte jetzt final, ohne Tools."}]})
            continue

        conv.append({"role": "user", "parts": response_parts})
        tool_only_turns += 1
        if tool_only_turns >= TOOL_TURN_LIMIT:
            conv.append({"role": "user", "parts": [{"text": "Fasse alles zusammen und antworte endgültig."}]})
            tool_only_turns = 0

    return "[Agent aborted: max_steps reached]"


async def _run_agent_local_tools(
    app: FastAPI,
    user_msg: str,
    max_steps: int,
    model: Optional[str],
    sid: str,
    provider: str,
) -> str:
    """
    ReAct-/JSON-Schema Loop für Modelle ohne native function calls (Ollama/OpenAI etc.).
    """
    conv = app.state.conversations.setdefault(sid, [])
    ctx = app.state.contexts.setdefault(sid, {})

    if not ctx.get("project_ref"):
        await ensure_project_index(app, sid, conv)

    # Build schema once
    tool_names = sorted(list(app.state.tools.keys()))
    schema = build_local_schema(tool_names)

    # System Prompt Zusatz: Erkläre das JSON-Protokoll
    protocol = (
        "Du MUSST IMMER im JSON-Format antworten (entspricht dem Schema). "
        "Wenn ein Tool nötig ist: {\"action\":\"tool\",\"tool\":\"<name>\",\"arguments\":{...}}. "
        "Wenn du fertig bist: {\"action\":\"final\",\"final\":\"<deine Antwort>\"}."
    )
    conv.insert(0, {"role": "user", "parts": [{"text": protocol}]})

    seen_calls: Set[str] = set()

    # Start mit User-Message
    conv.append({"role": "user", "parts": [{"text": user_msg}]})

    for _ in range(max_steps):
        # Call provider
        if provider == "openai":
            resp_raw = await llm_call_openai(conv, model)
            text = extract_text_gemini(resp_raw) or json.dumps(resp_raw)
        else:  # ollama
            resp_raw = await llm_call_ollama(conv, model, json_schema=schema)
            text = _ollama_to_gemini_like_text(resp_raw)

        # DeepSeek denkt mit <think>; entfernen
        text = strip_think(text)

        # JSON parsen
        try:
            data = json.loads(text)
        except Exception:
            # Modell hat Mist gebaut -> erneut erinnern
            conv.append({"role": "user", "parts": [{"text": "Bitte halte dich exakt an das JSON-Schema!"}]})
            continue

        action, tool, args, final_ans = parse_local_json(data)

        if action == "final":
            return final_ans or ""

        if action == "tool" and tool:
            sig = f"{tool}:{json.dumps(args, sort_keys=True)}"
            if sig in seen_calls:
                # Schleifen verhindern
                conv.append({"role": "user", "parts": [{"text": "Tool bereits ausgeführt. Gib bitte die finale Antwort."}]})
                continue
            seen_calls.add(sig)

            # Autofill/Call
            args = autofill_args(app, sid, tool, args)
            try:
                result = await call_mcp_tool(app, tool, args)
            except Exception as e:
                result = {"error": str(e)}
            update_context(app, sid, tool, result)

            # Observation zurück
            obs = json.dumps({"result": result}, ensure_ascii=False)
            conv.append({"role": "user", "parts": [{"text": f"OBSERVATION:\n{obs}\nDenke weiter und antworte wieder im JSON-Schema."}]})
            continue

        # Fallback
        conv.append({"role": "user", "parts": [{"text": "Unklare Aktion. Nutze 'tool' oder 'final'."}]})

    return "[Agent aborted: max_steps reached]"
