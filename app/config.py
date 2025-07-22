import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp import StdioServerParameters

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
SERVER_PATH = ROOT / "gitlab_mcp.py"

server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(SERVER_PATH)],
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    working_directory=str(ROOT),
)

# ---------------- LLM Settings ----------------
LLM_PROVIDER     = os.getenv("LLM_PROVIDER", "gemini").lower()  # gemini | openai | custom ...
DEFAULT_MODEL    = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")

# Gemini
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
GEMINI_TOOL_MODE = os.getenv("GEMINI_TOOL_MODE", "AUTO")  # AUTO | ANY | NONE
GEMINI_URL_TMPL  = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# OpenAI (optional)
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Custom (optional)
CUSTOM_LLM_BASE_URL = os.getenv("CUSTOM_LLM_BASE_URL", "")

# ---------------- System Prompt ----------------
SYSTEM_PROMPT = """
Du hast GitLab-Tools. Regeln:

A) Projektbezug
- Ist ein Projekt (ID/Name/Pfad) bereits im Kontext, nutze es direkt.
- Sonst: resolve_project(<name/pfad>) (toleriert Tippfehler). Bei eindeutigem Treffer (score ≥ 0.6) ohne Rückfrage verwenden.
- Für Inhalte (README, Dateien, Default-Branch): describe_project(<id>).

B) Dateien/Ordner
- Kennst du den default_branch nicht? -> describe_project.
- Struktur: list_repo_tree(path?, ref=default_branch, recursive=true)
- Datei: get_file(file_path, ref=default_branch)
- Schreiben: upsert_file(file_path, branch, content, commit_message)

C) Issues / MRs / CI
- Stelle sicher, dass project_ref gesetzt ist (siehe A).
- Verwende passende list_*/create_*/merge_*/trigger_* Tools.

D) Kontextpflege
- Nutze Fakten aus dem Verlauf (project_id, default_branch, aliases).
- Frage nicht erneut nach bereits bekannten Infos.

Antwortstil:
- Kompaktes Markdown (Listen, Tabellen, Codeblöcke).
- Kein Tool nötig? Antworte direkt.
"""
