from typing import Any, Dict, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    model: Optional[str] = None
    max_steps: int = 8
    reset: bool = False

class ToolCall(BaseModel):
    arguments: Dict[str, Any] = {}

class SelectProjectRequest(BaseModel):
    project_ref: str
    session_id: str = "default"
