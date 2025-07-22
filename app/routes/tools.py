from fastapi import APIRouter, HTTPException, Body, Request
from ..schemas import ToolCall
from ..mcp_client import call_mcp_tool

router = APIRouter()

@router.post("/tool/{tool}")
async def call_tool(tool: str, body: ToolCall = Body(...), request: Request = None):
    app = request.app
    try:
        data = await call_mcp_tool(app, tool, body.arguments)
        return {"result": data}
    except Exception as e:
        raise HTTPException(500, str(e))
