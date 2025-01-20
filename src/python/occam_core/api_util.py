from datetime import datetime
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel


class AgentInstanceMetadata(BaseModel):
    agent_instance_id: str


class AgentSetupErrorType(Enum):
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    INVALID_PARAMS = "INVALID_PARAMS"
    INSUFFICIENT_CREDITS = "INSUFFICIENT_CREDITS"
    INVALID_AGENT_INSTANCE_ID = "INVALID_AGENT_INSTANCE_ID"
    OTHER = "OTHER"


class AgentSetupError(BaseModel):
    error_type: AgentSetupErrorType
    error_message: str


class AgentRunDetail(BaseModel):
    agent_run_instance_id: str
    status: str
    start_time: datetime
    running_time_seconds: int
    # Placed here because we won't start with separate init and run methods.
    error: Optional[AgentSetupError] = None
