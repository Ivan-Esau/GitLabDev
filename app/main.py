from __future__ import annotations

import sys
import asyncio
from importlib import import_module
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import ROOT
from .mcp_client import init_mcp, close_mcp

# Windows Event-Loop Fix
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

app = FastAPI(title="MCP Studio")

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = ROOT / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.on_event("startup")
async def on_startup() -> None:
    await init_mcp(app)

@app.on_event("shutdown")
async def on_shutdown() -> None:
    await close_mcp(app)

API_PREFIX = "/api"
ROUTERS = (
    "routes.chat",
    "routes.projects",
    "routes.debug",
    "routes.coding",
    "routes.tools",
    "routes.local_llm",   # <--- neu
)
for mod_path in ROUTERS:
    try:
        mod = import_module(f"app.{mod_path}")
        app.include_router(mod.router, prefix=API_PREFIX)
    except Exception as e:
        print(f"[WARN] Router {mod_path} nicht geladen: {e}", file=sys.stderr)

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
