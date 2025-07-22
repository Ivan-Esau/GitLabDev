from fastapi import APIRouter, Query, Request

router = APIRouter()

@router.get("/debug/conv")
def debug_conv(request: Request, sid: str = Query("default")):
    return request.app.state.conversations.get(sid, [])

@router.get("/debug/context")
def debug_context(request: Request, sid: str = Query("default")):
    return request.app.state.contexts.get(sid, {})
