from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from occam_core.agents.model import AgentIdentityCoreModel, AgentIOModel
from occam_core.agents.util import AgentStatus
from pydantic import BaseModel, model_validator


class AgentCatalogueSDKResponse(BaseModel):
    connected: Dict[str, AgentIdentityCoreModel]
    unconnected: Dict[str, AgentIdentityCoreModel]

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


class AgentStateModel(AgentIOModel):
    """
    A unified agent IO model used to return agent responses.
    This covers
    1. Progress update (where chat_messages won't be set), status and timing data will be set.
    2. Error cases, where we return the error type
    3. Final output case, where chat_messages will be set.
    4. timestamp and duration of last run.
    5. batch numbers of chat_messages returned in the model. Assumption is that if we're returning
    multiple batches, the chat_messages from each batch will be merged into a single list.
    TODO: @medhat we might want to change the assumption in point 5, in which case we should preserve batch_numbers in each individual message
    saved in agent_output tables.
    """
    error_type: Optional[AgentHandlingErrorType] = None
    status: AgentStatus
    last_run_start_time: Optional[datetime] = None
    last_run_duration: Optional[int] = None
    chat_messages_batch_numbers: Optional[List[int]] = None

    # @medhat Suggestions for extensions.
    # messages_by_batch_number can be filtered by start_time
    # alive_intervals: Optional[List[Tuple[datetime, datetime]]] = None
    # running_intervals: Optional[List[Tuple[datetime, datetime]]] = None
    # messages_by_batch_number: Optional[Dict[int, List[OccamLLMMessage]]] = None

    @model_validator(mode="after")
    def validate_at_most_one_message_on_error(self):
        if self.error_type:
            assert len(self.chat_messages) <= 1

        return self


# errors and run details will both be structured as agent io models
# to provide a unified agent speech format, extra fields can either be
# conceptualized through special children of agnet io model, or using
# the extra field.





