from enum import Enum
from types import NoneType
from typing import Any, Dict, List, Optional, Type, Union

from occam_core.util.base_models import (AgentInstanceParamsModel,
                                         TAgentInstanceParamsModel)
from pydantic import BaseModel, Field

"""This file is auto-generated. Do not edit manually."""


class ChatPermissions(str, Enum):
    WRITE = "write"
    END_CHAT = "end_chat"


class SupervisionType(str, Enum):
    FULL = "full"
    SELECTIVE = "selective"
    FINAL = "final"
    BATCH = "batch"


class MailAction(str, Enum):
    SEND_MESSAGE = "send_message"
    GET_MESSAGE = "get_message"
    GET_THREAD = "get_thread"
    SEARCH_INBOX = "search_inbox"


class ChatMode(str, Enum):
    ROUND_ROBIN = "ROUND_ROBIN"
    GENERATIVE = "GENERATIVE"
    GROUP_CHAT = "GROUP_CHAT"
    WORKSPACE_CREATION = "WORKSPACE_CREATION"


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




class DefinedLLMAgentParamsModel(AgentInstanceParamsModel):
    system_prompt: Optional[str] = None
    log_chat: Optional[bool] = None
    assistant_name: Optional[str] = None
    retains_chat_history: bool = False
    response_format: Optional[type[BaseModel]] = None
    initial_chat_messages: Optional[list] = None


class MultiAgentWorkspaceCreatorParamsModel(AgentInstanceParamsModel):
    agents: Optional[dict[str, Union[TAgentInstanceParamsModel, NoneType]]] = None
    agent_turn_order: Optional[list[str]] = None
    session_id: Optional[str] = None
    max_chat_steps: int = 500
    chat_manager_name: str = "William of Occam"


class OccamProvidedUserAgentParamsModel(AgentInstanceParamsModel):
    session_id: Optional[str] = None
    chat_permissions: list[ChatPermissions] = Field(default_factory=lambda: None)


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


class MailToolAgentParamsModel(AgentInstanceParamsModel):
    action: MailAction
    action_email: str
    max_results: int = None


class MultiAgentWorkspaceParamsModel(AgentInstanceParamsModel):
    chat_goal: str = "Let's talk about all things that are good and lighthearted in the world."
    agent_selection_rule: ChatMode = ChatMode.ROUND_ROBIN
    agents: Optional[dict[str, Union[TAgentInstanceParamsModel, NoneType]]] = None
    agent_turn_order: Optional[list[str]] = None
    session_id: Optional[str] = None
    max_chat_steps: int = 500
    chat_manager_name: str = "William of Occam"


class SummarizerAgentParamsModel(AgentInstanceParamsModel):
    summary_length: int = 30
    custom_prompt: Optional[str] = None


class InvitedUserAgentParamsModel(AgentInstanceParamsModel):
    email: str
    first_name: str
    last_name: Optional[str] = None
    session_id: Optional[str] = None
    chat_permissions: list[ChatPermissions] = Field(default_factory=lambda: None)


class MultiAgentNetworkComputationAgentParamsModel(AgentInstanceParamsModel):
    session_id: Optional[str] = None
    network_goal: str = "process an input and produce an output"
    network_uuid: str
    max_steps: int = 500
    max_tokens: int = 100000
    max_retries: int = 3
    max_time: int = 3600


PARAMS_MODEL_CATALOGUE: Dict[str, Type[AgentInstanceParamsModel]] = {
    DefinedLLMAgentParamsModel.__name__: DefinedLLMAgentParamsModel,
    MultiAgentWorkspaceCreatorParamsModel.__name__: MultiAgentWorkspaceCreatorParamsModel,
    OccamProvidedUserAgentParamsModel.__name__: OccamProvidedUserAgentParamsModel,
    LLMAgentParamsModel.__name__: LLMAgentParamsModel,
    DataStructuringAgentParamsModel.__name__: DataStructuringAgentParamsModel,
    EmailCommunicatorAgentParamsModel.__name__: EmailCommunicatorAgentParamsModel,
    MailToolAgentParamsModel.__name__: MailToolAgentParamsModel,
    MultiAgentWorkspaceParamsModel.__name__: MultiAgentWorkspaceParamsModel,
    SummarizerAgentParamsModel.__name__: SummarizerAgentParamsModel,
    InvitedUserAgentParamsModel.__name__: InvitedUserAgentParamsModel,
    MultiAgentNetworkComputationAgentParamsModel.__name__: MultiAgentNetworkComputationAgentParamsModel,
}
