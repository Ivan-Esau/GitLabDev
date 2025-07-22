import json
import re
from typing import Any, Dict, List

def jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def try_json(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

def sanitize_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        props = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
        clean: Dict[str, Any] = {}

        t = schema.get("type")
        if t in {"object", "array", "string", "integer", "number", "boolean"}:
            clean["type"] = t

        if props:
            clean["properties"] = {k: sanitize_schema(v) for k, v in props.items()}

        if isinstance(schema.get("required"), list):
            req = [r for r in schema["required"] if r in props]
            if req:
                clean["required"] = req

        if "items" in schema:
            clean["items"] = sanitize_schema(schema["items"])

        if isinstance(schema.get("enum"), list):
            clean["enum"] = schema["enum"]
        if isinstance(schema.get("description"), str):
            clean["description"] = schema["description"]

        return clean or {"type": "object", "properties": {}}
    if isinstance(schema, list):
        return [sanitize_schema(x) for x in schema]
    return schema

def projects_md_table(projects: List[Dict[str, Any]]) -> str:
    lines = ["| ID | Name | Pfad | URL |", "|---|------|------|-----|"]
    for p in projects:
        lines.append(f"| {p.get('id')} | {p.get('name')} | {p.get('path')} | {p.get('web_url')} |")
    return "\n".join(lines)

_norm_re = re.compile(r"[^a-z0-9]")
def norm(s: str) -> str:
    return _norm_re.sub("", s.lower())
