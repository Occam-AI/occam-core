from enum import Enum
from typing import Any, Dict, List, Optional, Type

from occam_core.util.base_models import (AgentInstanceParamsModel,
                                         TAgentInstanceParamsModel)
from pydantic import BaseModel, Field

"""This file is auto-generated. Do not edit manually."""


class ChatPermissions(str, Enum):
    ANY = "any"


class SupervisionType(str, Enum):
    FULL = "full"
    SELECTIVE = "selective"
    FINAL = "final"
    BATCH = "batch"


class ChatSelectionRule(str, Enum):
    ROUND_ROBIN = "ROUND_ROBIN"
    GENERATIVE = "GENERATIVE"


class EmailCommunicatorCardModel(BaseModel):
    email: str
    first_name: str
    last_name: str
    company: str
    role: str


class SupervisorCardModel(BaseModel):
    email: str
    first_name: str
    last_name: str
    company: str
    role: str
    supervision_type: SupervisionType




class AgentChatCreatorParamsModel(AgentInstanceParamsModel):
    agents: Optional[dict[str, TAgentInstanceParamsModel]] = None
    agent_turn_order: Optional[list[str]] = None
    session_id: Optional[str] = None
    max_chat_steps: int = 1000
    chat_manager_name: str = "Occam Chat Assistant"


class DefinedLLMAgentParamsModel(AgentInstanceParamsModel):
    system_prompt: Optional[str] = None
    log_chat: Optional[bool] = None
    assistant_name: Optional[str] = None
    retains_chat_history: bool = False
    response_format: Optional[type[BaseModel]] = None
    initial_chat_messages: Optional[list] = None


class OccamProvidedUserAgentParamsModel(AgentInstanceParamsModel):
    session_id: Optional[str] = None
    chat_permission: ChatPermissions = ChatPermissions.ANY


class LLMAgentParamsModel(AgentInstanceParamsModel):
    system_prompt: Optional[str] = None
    multimodal_model_name: Optional[str] = None
    text_model_name: Optional[str] = None
    log_chat: Optional[bool] = None
    assistant_name: Optional[str] = None
    retains_chat_history: bool = False
    response_format: Optional[type[BaseModel]] = None
    initial_chat_messages: Optional[list] = None


class DataStructuringAgentParamsModel(AgentInstanceParamsModel):
    structuring_goal: str
    structured_output_model: Optional[str] = None


class EmailCommunicatorAgentParamsModel(AgentInstanceParamsModel):
    goal: str
    email_communicator_card: EmailCommunicatorCardModel
    supervisor_card: SupervisorCardModel


class AgentsChatParamsModel(AgentInstanceParamsModel):
    chat_goal: str = "Let's talk about all things that are good and lighthearted in the world."
    agent_selection_rule: ChatSelectionRule = ChatSelectionRule.ROUND_ROBIN
    agents: Optional[dict[str, TAgentInstanceParamsModel]] = None
    agent_turn_order: Optional[list[str]] = None
    session_id: Optional[str] = None
    max_chat_steps: int = 1000
    chat_manager_name: str = "Occam Chat Assistant"


class SummarizerAgentParamsModel(AgentInstanceParamsModel):
    summary_length: int = 30
    custom_prompt: Optional[str] = None


class InvitedUserAgentParamsModel(AgentInstanceParamsModel):
    email: str
    first_name: str
    last_name: Optional[str] = None
    session_id: Optional[str] = None
    chat_permission: ChatPermissions = ChatPermissions.ANY


PARAMS_MODEL_CATALOGUE: Dict[str, Type[AgentInstanceParamsModel]] = {
    AgentChatCreatorParamsModel.__name__: AgentChatCreatorParamsModel,
    DefinedLLMAgentParamsModel.__name__: DefinedLLMAgentParamsModel,
    OccamProvidedUserAgentParamsModel.__name__: OccamProvidedUserAgentParamsModel,
    LLMAgentParamsModel.__name__: LLMAgentParamsModel,
    DataStructuringAgentParamsModel.__name__: DataStructuringAgentParamsModel,
    EmailCommunicatorAgentParamsModel.__name__: EmailCommunicatorAgentParamsModel,
    AgentsChatParamsModel.__name__: AgentsChatParamsModel,
    SummarizerAgentParamsModel.__name__: SummarizerAgentParamsModel,
    InvitedUserAgentParamsModel.__name__: InvitedUserAgentParamsModel,
}
