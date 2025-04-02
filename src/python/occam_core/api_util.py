from datetime import datetime
from enum import Enum
from typing import Optional, Union

from occam_core.agents.model import AgentIOModel
from occam_core.enums import AgentRunStatus
from pydantic import BaseModel


class AgentHandlingErrorType(Enum):
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    INVALID_PARAMS = "INVALID_PARAMS"
    INSUFFICIENT_CREDITS = "INSUFFICIENT_CREDITS"
    INVALID_AGENT_INSTANCE_ID = "INVALID_AGENT_INSTANCE_ID"
    OTHER = "OTHER"
    # run errors
    AGENT_RUN_REQUEST_ERROR = "AGENT_RUN_REQUEST_ERROR"
    # add errors for pausing, resumign and terminating.


class AgentHandlingError(AgentIOModel):
    error_type: AgentHandlingErrorType
    error_message: str


class AgentResponseType(Enum):
    ERROR = "ERROR"
    UPDATE = "UPDATE"
    OUTPUT = "OUTPUT"


class AgentRunResponse(BaseModel):
   ...


class AgentResponseModel(AgentIOModel):
    # just ideas.
    # instance_id: str
    # result: Optional[AgentIOModel] = None
    response_type: AgentResponseType
    sub_type: Union[AgentHandlingErrorType, 'RunResponseType']
    error_response: Optional[str] = None
    run_response: Optional[AgentRunResponse] = None
    status: AgentRunStatus
    start_time: datetime
    running_time_seconds: int


# errors and run details will both be structured as agent io models
# to provide a unified agent speech format, extra fields can either be
# conceptualized through special children of agnet io model, or using
# the extra field.