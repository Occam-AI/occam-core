import inspect
from enum import Enum
from inspect import isabstract
from typing import Any, Dict, List, Optional, Self, Tuple, Type, TypeVar

from occam_core.agents.util import LLMIOModel, OccamLLMMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.parsed_chat_completion import (ParsedChatCompletion,
                                                      ParsedChoice)
from pydantic import BaseModel, Field, model_validator

from python.occam_core.agents.openai_models import ParsedFunctionToolCall


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
    WEB_TASKS = "WEB_TASKS"
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
    INPUT_TOKENS = "INPUT_TOKENS"
    OUTPUT_TOKENS = "OUTPUT_TOKENS"
    HOURLY_RATE = "HOURLY_RATE"
    WEB_SEARCH = "WEB_SEARCH"
    INTERNAL_REASONING = "INTERNAL_REASONING"
    IMAGE = "IMAGE"
    REQUESTS = "REQUESTS"
    INPUT_CACHE_READ = "INPUT_CACHE_READ"


price_type_to_macro_switch = {
    PriceTypes.INPUT_TOKENS: "${price_per_unit} / 1M",
    PriceTypes.OUTPUT_TOKENS: "${price_per_unit} / 1M",
    PriceTypes.WEB_SEARCH: "${price_per_unit} / 1M",
    PriceTypes.INTERNAL_REASONING: "${price_per_unit} / 1M",
    PriceTypes.IMAGE: "${price_per_unit} / 1M",
    PriceTypes.INPUT_CACHE_READ: "${price_per_unit} / 1M",
    PriceTypes.HOURLY_RATE: "${price_per_unit} / hour",
    PriceTypes.REQUESTS: "${price_per_unit} / request",
}

price_type_to_sort_order = {
    PriceTypes.INPUT_TOKENS: 1,
    PriceTypes.OUTPUT_TOKENS: 2,
    PriceTypes.WEB_SEARCH: 3,
    PriceTypes.REQUESTS: 4,
    PriceTypes.IMAGE: 5,
    PriceTypes.INTERNAL_REASONING: 6,
    PriceTypes.INPUT_CACHE_READ: 7,
    PriceTypes.HOURLY_RATE: 8,
}


class AgentPriceModel(BaseModel):
    type_: PriceTypes
    minimum_charge: Optional[float] = None
    price_display: Optional[str] = None
    price_per_unit: float

    @model_validator(mode="after")
    def set_price_display(self) -> Self:
        if self.price_display is not None:
            return self
        if (self.price_per_unit or 0) == 0:
            self.price_display = "FREE"
        else:
            self.price_display = price_type_to_macro_switch[self.type_].format(
                price_per_unit=self.price_per_unit
            )
        return self


class AgentPriceModels(BaseModel):
    models: List[AgentPriceModel]

    @model_validator(mode="after")
    def set_price_display(self) -> Self:
        new_models: List[AgentPriceModel] = []
        for model in self.models:
            if model.type_ != PriceTypes.HOURLY_RATE or model.price_display != "FREE":
                new_models.append(model)
        new_models.sort(key=lambda x: price_type_to_sort_order[x.type_])
        self.models = new_models
        return self


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
    price_models: AgentPriceModels = Field(default_factory=lambda: AgentPriceModels(models=[]))
    is_bot: bool = True

    # for loading the params model
    # needed for instantiating the agent with.
    instance_params_model_name: str
    is_ready_to_run: bool


chat_completion_fields = set(ChatCompletion.model_fields.keys())
llm_model_fields = set(LLMIOModel.model_fields.keys())
intersecting_llm_model_fields = llm_model_fields.intersection(chat_completion_fields)

choice_fields = set(Choice.model_fields.keys())
llm_message_fields = set(OccamLLMMessage.model_fields.keys())
intersecting_choice_fields = choice_fields.intersection(llm_message_fields)


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

    @classmethod
    def from_llm_response(
        cls,
        llm_response: ChatCompletion | ParsedChatCompletion,
        assistant_name: str,
    ) -> Self:
        """
        Convert LLM response to AgentIOModel.

        This method creates an AgentIOModel from ChatCompletion or 
        ParsedChatCompletion responses, which is the core model of our 
        architecture.

        Key transformations:
        - Choice-level fields are moved to the message level in our model
        - Selected ChatCompletion fields are included in the AgentIOModel
        - Assistant name is assigned since it's not returned by the LLM

        Args:
            llm_response: ChatCompletion or ParsedChatCompletion response
            assistant_name: Name to assign to the assistant messages

        Returns:
            AgentIOModel with converted messages and metadata

        """
        response_models, tool_calls = \
            AgentIOModel.pop_structured_outputs(llm_response)
        llm_response_model = llm_response.model_dump(mode="json")
        AgentIOModel.re_add_structured_outputs(
            llm_response_model,
            response_models,
            tool_calls
        )

        init_variables = {}
        for field in intersecting_llm_model_fields:
            init_variables[field] = llm_response_model.get(field)


        messages = []
        for choice in llm_response_model.get("choices", []):
            message_init_variables = {}

            # we get top level choice fields that we use at message level.
            # basically logprobs and finish_reason.
            for field in intersecting_choice_fields:
                message_init_variables[field] = choice.get(field)

            message: dict = choice.get("message", {})
            # we get all fields since occam message inherits from chat completion message.
            for field in message.keys():
                message_init_variables[field] = message.get(field)

            occam_message = OccamLLMMessage.model_validate(message_init_variables)
            occam_message.name = assistant_name
            messages.append(occam_message)
        agent_model = cls.model_validate(init_variables)
        agent_model.chat_messages = messages
        return agent_model

    @staticmethod
    def pop_structured_outputs(llm_response: ChatCompletion | ParsedChatCompletion) \
        -> Tuple[Dict[int, BaseModel], Dict[int, List[ParsedFunctionToolCall]]]:
        """
        this is to preserve pydantic models for structured outputs
        and tool calls, instead of dumping then failing to validate
        back.
        """

        response_models = {}
        tool_calls = {}
        for choice in llm_response.choices:
            if choice.message.parsed:
                response_models[choice.index] = choice.message.parsed
                choice.message.parsed = None
            if choice.message.tool_calls:
                tool_calls[choice.index] = choice.message.tool_calls
                choice.message.tool_calls = None
        return response_models, tool_calls

    @staticmethod
    def re_add_structured_outputs(llm_response_model: dict, response_models: dict, tool_calls: dict) -> None:
        """
        this is to re-add the structured outputs to the llm_response_model
        """
        # we need to preserve pydantic models for structured outputs
        # so we override the dumps with them.
        for choice in llm_response_model.get("choices", []):
            if choice.get("index") in response_models:
                choice["message"]["parsed"] = response_models[choice["index"]]
            if choice.get("index") in tool_calls:
                choice["message"]["tool_calls"] = tool_calls[choice["index"]]


TAgentIOModel = TypeVar("TAgentIOModel", bound=AgentIOModel)
