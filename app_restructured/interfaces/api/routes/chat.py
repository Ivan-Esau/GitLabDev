from fastapi import APIRouter, HTTPException, Request
import httpx

from app_restructured.interfaces.api.schemas.schemas import ChatRequest
from app_restructured.application.services.agent import run_agent

router = APIRouter()

@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    app = request.app
    if req.reset:
        app.state.conversations[req.session_id] = []
        app.state.contexts[req.session_id] = {}
    try:
        answer = await run_agent(
            app,
            req.message,
            req.max_steps,
            req.model,
            req.session_id,
            provider=req.provider,
        )
        return {"answer": answer}
    except httpx.HTTPStatusError as e:
        return {"error": e.response.text}
    except Exception as e:
        raise HTTPException(500, str(e))
