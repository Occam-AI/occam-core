import enum
from typing import List, Optional, Type

from pydantic import BaseModel, model_validator

from python.occam_core.util.base_models import ParamsIOModel


class CommunicationMethod(str, enum.Enum):
    SLACK = "slack"
    EMAIL = "email"


class ChatChannelPermission(enum.Enum):
    READ_ONLY = "read_only"
    SEND_MESSAGE = "send_message"
    UPLOAD_FILES = "upload_files"
    TERMINATE_CHAT = "terminate_chat"
    ALL = "ALL"



class UserAgentParamsModel(ParamsIOModel):
    user_id: int
    email: str
    first_name: str
    last_name: str
    slack_handle: Optional[str] = None
    channel_permission: ChatChannelPermission = ChatChannelPermission.READ_ONLY
    # which methods are we allowed to reach user through.
    communication_methods: Optional[List[CommunicationMethod]] = None

    @model_validator(mode="after")
    def check_either_slack_or_email(cls, v):
        if CommunicationMethod.SLACK in v.communication_methods:
            assert v.slack_handle, "Slack handle must be provided"
        if CommunicationMethod.SLACK and v.slack_handle is None:
            raise ValueError("Slack handle must be provided given that slack communication method is selected.")
        return v

    @model_validator(mode="after")
    def check_communication_methods(cls, v):
        if v.communication_methods is None:
            v.communication_methods = [CommunicationMethod.EMAIL]
        return v


class LLMParamsModel(ParamsIOModel):
    system_prompt: Optional[str] = None
    llm_model_name: Optional[str] = None
    image_model_name: Optional[str] = None
    log_chat: Optional[bool] = None
    assistant_name: Optional[str] = None
    # non-serializable for a plan.
    response_format: Optional[Type[BaseModel]] = None


class SupervisionType(str, enum.Enum):
    FULL = "full"
    SELECTIVE = "selective"
    FINAL = "final"
    BATCH = "batch"


class EmailCommunicatorCardModel(BaseModel):
    """
    A card that contains information about the person
    that the email communicator bot is speaking on behalf of.
    """

    email: str
    first_name: str
    last_name: str
    company: str
    role: str


class SupervisorCardModel(EmailCommunicatorCardModel):
    supervision_type: SupervisionType


class EmailCommunicatorParamsModel(ParamsIOModel):
    """
    Parameters for the email communicator tool.
    """

    goal: str
    email_communicator_card: EmailCommunicatorCardModel
    supervisor_card: SupervisorCardModel
