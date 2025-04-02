from datetime import datetime
from enum import Enum
from typing import Optional

from occam_core.agents.model import AgentIOModel
from occam_core.enums import AgentRunStatus
from pydantic import model_validator


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


class AgentResponseModel(AgentIOModel):
    response_type: AgentResponseType
    error_type: Optional[AgentHandlingErrorType] = None
    status: AgentRunStatus
    start_time: datetime
    running_time_seconds: int

    @model_validator(mode="after")
    def validate_single_message_on_error(self):
        if self.error_type:
            assert len(self.chat_messages) == 1
            assert self.chat_messages[0].role == "assistant"

        return self


# errors and run details will both be structured as agent io models
# to provide a unified agent speech format, extra fields can either be
# conceptualized through special children of agnet io model, or using
# the extra field.