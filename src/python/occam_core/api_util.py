from datetime import datetime
from enum import Enum
from typing import Optional, Union

from occam_core.agents.model import AgentIOModel
from occam_core.enums import AgentRunStatus
from pydantic import BaseModel


class AgentInstanceMetadata(BaseModel):
    agent_instance_id: str


class AgentHandlingErrorType(Enum):
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    INVALID_PARAMS = "INVALID_PARAMS"
    INSUFFICIENT_CREDITS = "INSUFFICIENT_CREDITS"
    INVALID_AGENT_INSTANCE_ID = "INVALID_AGENT_INSTANCE_ID"
    OTHER = "OTHER"
    # run errors
    AGENT_RUN_REQUEST_ERROR = "AGENT_RUN_REQUEST_ERROR"
    # add errors for pausing, resumign and terminating.


class AgentHandlingError(BaseModel):
    error_type: AgentHandlingErrorType
    error_message: str


class AgentRunDetail(BaseModel):
    instance_id: str
    result: Optional[AgentIOModel] = None
    status: AgentRunStatus
    start_time: datetime
    running_time_seconds: int
