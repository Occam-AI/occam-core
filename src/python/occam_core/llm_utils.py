import enum
from typing import Any, Dict, Optional, Type, TypeVar

from occam_core.base_models import IOModel, ParamsIOModel
from occam_core.util.data_types.occam_data_type import OccamDataType
from occam_core.util.functions import remove_extra_spaces
from pydantic import BaseModel, field_validator, model_validator

IBaseModel = TypeVar('IBaseModel', bound=BaseModel)


class OpenAIModelEnum(str, enum.Enum):
    GPT3_5 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    O1_PREVIEW = "o1-preview"
    GPT4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    GPT_4_VISION_PREVIEW = "gpt-4-vision-preview"
    GPT4_O = "gpt-4o"
    GPT4_O_beta = "gpt-4o-2024-08-06"


class OpenRouterModel(str, enum.Enum):
    Auto = "openrouter/auto"
    MythoMax_13B = "gryphe/mythomax-l2-13b"
    MythoMax_13B_8k = "gryphe/mythomax-l2-13b-8k"
    GPT_4_Turbo = "openai/gpt-4-1106-preview"
    Mistral_7B_Instruct = "mistralai/mistral-7b-instruct"
    HuggingFace_Zephyr_7B = "huggingfaceh4/zephyr-7b-beta"
    OpenChat_3_5 = "openchat/openchat-7b"
    MythoMist_7B = "gryphe/mythomist-7b"
    Cinematika_7B_alpha = "openrouter/cinematika-7b"
    RWKV_v5_World_3B_beta = "rwkv/rwkv-5-world-3b"
    RWKV_v5_3B_AI_Town_beta = "recursal/rwkv-5-3b-ai-town"
    Psyfighter_13B = "jebcarter/psyfighter-13b"
    Psyfighter_v2_13B = "koboldai/psyfighter-13b-2"
    Noromaid_Mixtral_8x7B_Instruct = "neversleep/noromaid-mixtral-8x7b-instruct"
    Nous_Hermes_13B = "nousresearch/nous-hermes-llama2-13b"
    Meta_CodeLlama_34B_Instruct = "meta-llama/codellama-34b-instruct"
    Phind_CodeLlama_34B_v2 = "phind/phind-codellama-34b"
    Neural_Chat_7B_v3_1 = "intel/neural-chat-7b"
    Mistral_Mixtral_8x7B_Instruct_beta = "mistralai/mixtral-8x7b-instruct"
    Llava_13B = "haotian-liu/llava-13b"
    Nous_Hermes_2_Vision_7B_alpha = "nousresearch/nous-hermes-2-vision-7b"
    Meta_Llama_v2_13B_Chat = "meta-llama/llama-2-13b-chat"
    OpenAI_GPT_3_5_Turbo = "openai/gpt-3.5-turbo"
    OpenAI_GPT_3_5_Turbo_16k_preview = "openai/gpt-3.5-turbo-1106"
    OpenAI_GPT_3_5_Turbo_16k = "openai/gpt-3.5-turbo-16k"
    OpenAI_GPT_4_Turbo_preview = "openai/gpt-4-1106-preview"
    OpenAI_GPT_4 = "openai/gpt-4"
    OpenAI_GPT_4_32k = "openai/gpt-4-32k"
    OpenAI_GPT_4_Vision_preview = "openai/gpt-4-vision-preview"
    OpenAI_GPT_3_5_Turbo_Instruct = "openai/gpt-3.5-turbo-instruct"
    Google_PaLM_2_Chat = "google/palm-2-chat-bison"
    Google_PaLM_2_Code_Chat = "google/palm-2-codechat-bison"
    Google_PaLM_2_Chat_32k = "google/palm-2-chat-bison-32k"
    Google_PaLM_2_Code_Chat_32k = "google/palm-2-codechat-bison-32k"
    Google_Gemini_Pro_preview = "google/gemini-pro"
    Google_Gemini_Pro_Vision_preview = "google/gemini-pro-vision"
    Perplexity_PPLX_70B_Online = "perplexity/pplx-70b-online"
    Perplexity_PPLX_7B_Online = "perplexity/pplx-7b-online"
    Perplexity_PPLX_7B_Chat = "perplexity/pplx-7b-chat"
    Perplexity_PPLX_70B_Chat = "perplexity/pplx-70b-chat"
    Meta_Llama_v2_70B_Chat = "meta-llama/llama-2-70b-chat"
    Nous_Hermes_70B = "nousresearch/nous-hermes-llama2-70b"
    Nous_Capybara_34B = "nousresearch/nous-capybara-34b"
    Airoboros_70B = "jondurbin/airoboros-l2-70b"
    Synthia_70B = "migtissera/synthia-70b"
    OpenHermes_2_Mistral_7B = "teknium/openhermes-2-mistral-7b"
    OpenHermes_2_5_Mistral_7B = "teknium/openhermes-2.5-mistral-7b"
    Pygmalion_Mythalion_13B = "pygmalionai/mythalion-13b"
    ReMM_SLERP_13B = "undi95/remm-slerp-l2-13b"
    Xwin_70B = "xwin-lm/xwin-lm-70b"
    Toppy_M_7B = "undi95/toppy-m-7b"
    Goliath_120B = "alpindale/goliath-120b"
    lzlv_70B = "lizpreciatior/lzlv-70b-fp16-hf"
    Noromaid_20B = "neversleep/noromaid-20b"
    Yi_34B_Chat = "01-ai/yi-34b-chat"
    Yi_34B_base = "01-ai/yi-34b"
    Yi_6B_base = "01-ai/yi-6b"
    StripedHyena_Nous_7B = "togethercomputer/stripedhyena-nous-7b"
    StripedHyena_Hessian_7B_base = "togethercomputer/stripedhyena-hessian-7b"
    Mistral_Mixtral_8x7B_base = "mistralai/mixtral-8x7b"
    Dolphin_2_6_Mixtral_8x7B = "cognitivecomputations/dolphin-mixtral-8x7b"


class AnthropicModel(str, enum.Enum):
    Claude3_Opus = 'claude-3-opus-20240229'
    Claude3_Sonnet = 'claude-3-sonnet-20240229'
    Claude3_Haiku = 'claude-3-haiku-20240307'


class LLMRole(str, enum.Enum):
    assistant = "assistant"
    system = "system"
    user = "user"


class OccamLLMMessage(BaseModel):
    content: str | list[dict[str, Any]]
    role: LLMRole # system vs user vs assistant
    name: Optional[str] = None # this allows us to distinguish users in a multi-user chat
    # FIXME: note this is a pydantic model of the sturcture output if available.
    # at present this means that we can't load this structured output back
    # from the DB as we don't know what the model is.
    # even if we store it as IBaseModel, we have the same issue.
    parsed: Optional[Any] = None

    @model_validator(mode="after")
    def format_messenger_name(cls, v):
        v.name = format_llm_messenger_name(v.name)
        return v


class OccamLLMChatHistory(OccamDataType):

    system_prompt: str
    system_prompt_as_message: Optional[OccamLLMMessage] = None
    messages: Optional[list[OccamLLMMessage]] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.post_init()

    def post_init(self):
        self.system_prompt_as_message = OccamLLMMessage(
            content=self.system_prompt,
            role=LLMRole.user,
            name="system"
        )
        if not self.messages:
            self.messages = []

    def override_messages(self, messages: list[OccamLLMMessage]):
        self.messages = messages

    def extend_messages(self, messages: list[OccamLLMMessage]) -> None:
        self.messages.extend(messages)

    def delete_messages(self) -> None:
        self.messages = []

    def add_message(self, content: str | list[dict[str, Any]], role: LLMRole, name: Optional[str] = None, prepend: bool = False):
        message = OccamLLMMessage(content=content, role=role, name=name)
        self.messages.insert(len(self.messages) if not prepend else 0, message)

    def model_dump_history(self) -> list[dict[str, Any]]:
        return [self.system_prompt_as_message.model_dump()] + [message.model_dump() for message in self.messages]

    def get_history(self):
        return [self.system_prompt_as_message] + self.messages

    def print(self):
        for i, message in enumerate(self.get_history()):
            print(f"message number: {i}, role: {message.role}, name: {message.name}, content: {message.content}")

    @staticmethod
    def message_to_str(message_index: int, message: OccamLLMMessage) -> str:
        return ", ".join([
            f"message number: {message_index}",
            f"role: {message.role}",
            f"name: {message.name}",
            f"content: {message.content}"
        ])

    def history_to_str(self) -> str:
        return "\n".join([self.message_to_str(i, message) for i, message in enumerate(self.get_history())])


class LLMInputModel(IOModel):
    """
    Input model for LLM tools.

    chat messages append first, then prompt then query.

    Rationale is group agent chat settings, where there's a shared
    context of messages, and a prompt and a query, whose content
    is hidden from the conversationalists, before and after.

    eg. an agent selector llm, is fed with a query to pick next
    agent in a conversation, with the convo being the chat messages
    list.
    """
    chat_messages: Optional[list[OccamLLMMessage]] = None

    # intermediate prompt that can be used to guide interpretation
    # of the message to follow.
    prompt: Optional[str] = None
    # llm expects either a single query or a list of chat messages
    query: Optional[str] = None

    # reset chat messages means that if this rool has been running
    # a chat prior ton this run call, we ignore all previous convos
    # and only factor in the initial system prompt of the LLM.
    reset_chat_messages: bool = False

    # role of the user providing the query or chat messages
    role: LLMRole = LLMRole.user

    # name of the user, eg. engineer, architect etc.
    name: Optional[str] = None

    # file paths to be used for vision models
    file_paths: Optional[list[str]] = None

    # we dont' save as TypeVar IBaseModel as it can't be loaded back and beaks validation
    # as a result.
    # ASSUMPTION: response format is ONLY provided through a direct instasntiation
    # inside composite tools i.e. LLMInputModel(response_format=)
    # NOTE model_validate_json, as we won't know how to load it.
    response_format: Optional[Any] = None

    @field_validator('query', 'prompt', mode="before")
    @classmethod
    def transform(cls, raw: str) -> tuple[int, int]:
        if isinstance(raw, str):
            raw = remove_extra_spaces(raw)
        return raw


class LLMParamsModel(ParamsIOModel):
    system_prompt: Optional[str] = None
    llm_model_name: Optional[str] = None
    image_model_name: Optional[str] = None
    log_chat: Optional[bool] = None
    assistant_name: Optional[str] = None
    # NOTE: this is currently not used/tested and will
    # break serialization.
    response_format: Optional[Type[BaseModel]] = None


class OccamLLMChoice(BaseModel):
    finish_reason: str
    message: OccamLLMMessage


class OccamLLMUsage(BaseModel):
    total_tokens: int
    completion_tokens: int
    prompt_tokens: int


class OccamLLMResponse(OccamDataType):
    id: str
    choices: list[OccamLLMChoice]
    model: str
    usage: OccamLLMUsage
    # FIXME: shouldn't be an output. doesn't matter now cause we're using it internally.
    structured_responses: list[IBaseModel]


class LLMOutputModel(IOModel, OccamLLMResponse):
    text: str
    chat_history: OccamLLMChatHistory


def process_llm_response(
        llm_response_dict: Dict[str, Any],
        response_format: IBaseModel = None,
        assistant_name: str = None) -> OccamLLMResponse:
    # Converting any OpenAIObjects to dicts
    structured_responses = []
    llm_response_dict["choices"] = [dict(d) for d in llm_response_dict["choices"]]
    for d in llm_response_dict["choices"]:
        d["message"] = dict(d["message"])
        d["message"]["name"] = assistant_name
        if response_format:
            structured_responses.append(d["message"]["parsed"])

    llm_response_dict["usage"] = dict(llm_response_dict["usage"])
    llm_response_dict["structured_responses"] = structured_responses
    return OccamLLMResponse.model_validate(llm_response_dict)


def format_llm_messenger_name(name: str):
    if name:
        return name.replace(" ", "")


def construct_response_format_model_from_io_model(response_model_name: str, io_model: Type[IOModel]) -> Type[OccamDataType]:

    # Get annotations and defaults from the IOModel
    annotations: Dict[str, Type[Any]] = io_model.get_variable_types_map()

    # Combine annotations and defaults into a dictionary for the model creation
    model_attributes = {"__annotations__": annotations}
    # model_attributes.update(defaults)

    # Dynamically create a Pydantic model with the given annotations and defaults
    model = type(response_model_name, (OccamDataType,), model_attributes)

    return model
