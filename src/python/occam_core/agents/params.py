import enum
from typing import List, Optional, Type

from occam_core.util.base_models import AgentInstanceParamsModel
from pydantic import BaseModel, model_validator


class CommunicationMethod(str, enum.Enum):
    SLACK = "slack"
    EMAIL = "email"


class ChatChannelPermission(enum.Enum):
    READ_ONLY = "read_only"
    SEND_MESSAGE = "send_message"
    UPLOAD_FILES = "upload_files"
    TERMINATE_CHAT = "terminate_chat"
    ALL = "ALL"


# User Provided Human Agent Params
class UserProvidedHumanAgentParamsModel(AgentInstanceParamsModel):
    """
    This is when you need to bring in a human agent into a workflow
    using agents chat, where the human agent is provided by the user
    in full.
    """

    email: str
    first_name: str
    last_name: str
    slack_handle: Optional[str] = None
    channel_permission: ChatChannelPermission = ChatChannelPermission.READ_ONLY
    communication_methods: Optional[List[CommunicationMethod]] = None

    @model_validator(mode="after")
    def check_either_slack_or_email(self):
        if CommunicationMethod.SLACK in self.communication_methods:
            assert self.slack_handle, "Slack handle must be provided"
        return self


# Occam Provided Human Agent Params
class OccamProvidedHumanAgentParamsModel(AgentInstanceParamsModel):
    """
    This is for occam provided interface agents for which we
    have contact information and only need to know their designated
    permissions and communication methods.
    """

    channel_permission: ChatChannelPermission = ChatChannelPermission.READ_ONLY
    # which methods are we allowed to reach user through.
    communication_methods: Optional[List[CommunicationMethod]] = None

    @model_validator(mode="after")
    def check_communication_methods(self):
        if self.communication_methods is None:
            self.communication_methods = [CommunicationMethod.EMAIL]
        return self


# LLM Agent Params
class LLMAgentParamsModel(AgentInstanceParamsModel):
    llm_model_name: str
    system_prompt: Optional[str] = None
    image_model_name: Optional[str] = None
    log_chat: Optional[bool] = None
    assistant_name: Optional[str] = None


class DefinedLLMAgentParamsModel(AgentInstanceParamsModel):
    system_prompt: Optional[str] = None
    image_model_name: Optional[str] = None
    log_chat: Optional[bool] = None
    assistant_name: Optional[str] = None


# Communication Agent Params
class SupervisionType(str, enum.Enum):
    FULL = "full"
    SELECTIVE = "selective"
    FINAL = "final"
    BATCH = "batch"


class EmailCommunicatorCardModel(BaseModel):

    email: str
    first_name: str
    last_name: str
    company: str
    role: str


class SupervisorCardModel(EmailCommunicatorCardModel):
    supervision_type: SupervisionType


class EmailCommunicatorAgentParamsModel(AgentInstanceParamsModel):
    """
    Parameters for the email communicator tool.
    """

    goal: str
    email_communicator_card: EmailCommunicatorCardModel
    supervisor_card: SupervisorCardModel
