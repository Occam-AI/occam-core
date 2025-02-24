import inspect
from enum import Enum
from inspect import isabstract
from typing import Any, Dict, List, Optional, Type, TypeVar

from occam_core.agents.params import PARAMS_MODEL_CATALOGUE
from occam_core.agents.util import LLMInputModel, OccamLLMMessage
from occam_core.util.base_models import IOModel
from pydantic import BaseModel, model_validator


class AgentType(str, Enum):
    Occam_AI_Agent = "Occam_AI_Agent"
    Browser_Based_AI_Agent = "Browser_Based_AI_Agent"
    Human = "Human"
    External_Worker = "External_Worker"


class AgentRole(str, Enum):
    GENERAL = "general"
    LANGUAGE_MODEL = "language_model"
    todo = "todo"


class AgentIdentityCoreModel(BaseModel):
    """
    This model enforces field requirements based on AgentType.
    """

    # name information.
    key: str
    # some agents' name is based on key, for humans, it's first and last name.
    name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None

    # agent contact information
    email: Optional[str] = None
    slack_handle: Optional[str] = None

    # agent properties
    type: AgentType
    role: AgentRole
    role_embedding_vector: Optional[List[float]] = None
    short_description: Optional[str] = None
    long_description: Optional[str] = None
    price_per_input_token: Optional[float] = None
    price_per_output_token: Optional[float] = None
    price_per_response: Optional[float] = None
    is_bot: bool = True

    # for loading the params model
    # needed for instantiating the agent with.
    instance_params_model_name: str

    @model_validator(mode="after")
    def validate_params_model_name(self):
        if self.instance_params_model_name not in PARAMS_MODEL_CATALOGUE:
            raise ValueError(f"agent {self.name}'s params model {self.instance_params_model_name} not found in params_model catalogue.")
        return self


class AgentIOModel(LLMInputModel):
    extra: Optional[Any] = None

    @property
    def last_message(self) -> OccamLLMMessage:
        return self.chat_messages[-1]


TAgentIOModel = TypeVar("TAgentIOModel", bound=AgentIOModel)
