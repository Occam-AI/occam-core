import inspect
from enum import Enum
from inspect import isabstract
from typing import Any, Dict, List, Optional, Self, Type, TypeVar

from occam_core.agents.util import LLMIOModel, OccamLLMMessage
from occam_core.util.base_models import AgentInstanceParamsModel
from pydantic import BaseModel, Field, model_validator


class AgentType(str, Enum):
    Occam_AI_Agent = "Occam_AI_Agent"
    Browser_Based_AI_Agent = "Browser_Based_AI_Agent"
    Human = "Human"
    External_Worker = "External_Worker"


class OccamCategory(str, Enum):
    GENERAL = "GENERAL"
    LANGUAGE_MODEL = "LANGUAGE_MODEL"
    VISION_MODEL = "VISION_MODEL"
    REASONING_MODEL = "REASONING_MODEL"
    DATA_MANIPULATION = "DATA_MANIPULATION"
    COMMUNICATION = "COMMUNICATION"
    WEB_SEARCH = "WEB_SEARCH"
    ACCOUNTING_FINANCE = "ACCOUNTING_FINANCE"
    HUMAN_RESOURCES = "HUMAN_RESOURCES"
    SCIENCE = "SCIENCE"
    ANALYSIS = "ANALYSIS"
    USER_AGENT = "USER_AGENT"
    FILE_SYSTEM = "FILE_SYSTEM"
    DATABASE = "DATABASE"
    WORKSPACE = "WORKSPACE"



class PriceTypes(str, Enum):
    INPUT_TOKEN = "INPUT_TOKEN"
    OUTPUT_TOKEN = "OUTPUT_TOKEN"
    HOUR = "HOUR"


class PriceModel(BaseModel):
    type_: PriceTypes
    unit: str
    minimum_charge: float
    macro: str = "{unit}"
    price_per_unit: float


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
    category: OccamCategory
    agent_embedding_vector: Optional[List[float]] = None
    short_description: Optional[str] = None
    long_description: Optional[str] = None
    price_per_input_token: Optional[float] = None
    price_per_output_token: Optional[float] = None
    price_per_hour: Optional[float] = None
    minimum_charge: Optional[float] = None
    #price_model: Optional[PriceModel] = None
    is_bot: bool = True

    # for loading the params model
    # needed for instantiating the agent with.
    instance_params_model_name: str
    is_ready_to_run: bool


class AgentIOModel(LLMIOModel):
    """
    IO model for agents.
    """

    extra: Optional[Any] = None
    _text: Optional[str] = None

    @property
    def last_message(self) -> OccamLLMMessage:
        return self.chat_messages[-1] if self.chat_messages else None

    @property
    def text(self) -> str:
        if not self._text:
            self._text = '\n'.join([message.content for message in self.chat_messages])
        return self._text

    @classmethod
    def merge_models(cls, models: List[Self]) -> Self:
        """
        Merge a list of AgentIOModel objects into a single AgentIOModel object.

        Note: this assumes models only have chat messages
        """
        merged_model = cls()
        for model in models:
            merged_model.chat_messages.extend(model.chat_messages)
        return merged_model


TAgentIOModel = TypeVar("TAgentIOModel", bound=AgentIOModel)
