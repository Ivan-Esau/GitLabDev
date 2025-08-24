from typing import Any, Dict, Optional, Union
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    model: Optional[str] = None
    provider: Optional[str] = None   # <--- neu
    max_steps: int = 8
    reset: bool = False

class ToolCall(BaseModel):
    arguments: Dict[str, Any] = {}

class SelectProjectRequest(BaseModel):
    project_ref: Union[str, int]
    session_id: str = "default"
