import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .routes import chat, projects, tools, debug, coding

from .mcp_client import init_mcp, close_mcp
from .routes import chat, projects, tools, debug

log = logging.getLogger("uvicorn.error")

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

app = FastAPI()

# Static frontend
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))

# API routes
app.include_router(chat.router,     prefix="/api")
app.include_router(projects.router, prefix="/api")
app.include_router(tools.router,    prefix="/api")
app.include_router(debug.router,    prefix="/api")
app.include_router(coding.router, prefix="/api")

@app.on_event("startup")
async def on_startup():
    await init_mcp(app)
    # Coding-Agent Status Speicher
    app.state.coding_status = {}
    log.info("Application startup complete.")


@app.on_event("shutdown")
async def on_shutdown():
    await close_mcp(app)
