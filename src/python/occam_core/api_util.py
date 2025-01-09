from enum import Enum
from typing import Union

from pydantic import BaseModel


class AgentInstanceMetadata(BaseModel):
    agent_instance_id: str


class AgentFetchErrorType(Enum):
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    OTHER = "OTHER"


class AgentFetchError(BaseModel):
    error_type: AgentFetchErrorType
    error_message: str


class AgentInstantiationErrorType(Enum):
    INVALID_PARAMS = "INVALID_PARAMS"
    INSUFFICIENT_CREDITS = "INSUFFICIENT_CREDITS"
    OTHER = "OTHER"


class AgentInstantiationError(BaseModel):
    error_type: AgentInstantiationErrorType
    error_message: str


class AgentInstanceFetchErrorType(Enum):
    INVALID_AGENT_INSTANCE_ID = "INVALID_AGENT_INSTANCE_ID"
    OTHER = "OTHER"


class AgentInstanceFetchError(BaseModel):
    error_type: AgentInstanceFetchErrorType
    error_message: str


class AgentRunDetail(BaseModel):
    status: str
