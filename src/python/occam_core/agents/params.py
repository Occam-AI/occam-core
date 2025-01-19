import enum
from typing import List, Optional, Type

from occam_core.agents.util import ChatChannelPermission, CommunicationMethod
from occam_core.util.base_models import AgentInstanceParamsModel, ParamsIOModel
from pydantic import BaseModel, model_validator


class UserAgentParticipationParamsModel(AgentInstanceParamsModel):

    channel_permission: ChatChannelPermission = ChatChannelPermission.READ_ONLY
    # which methods are we allowed to reach user through.
    communication_methods: Optional[List[CommunicationMethod]] = None

    @model_validator(mode="after")
    def check_communication_methods(self):
        if self.communication_methods is None:
            self.communication_methods = [CommunicationMethod.EMAIL]
        return self


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
