import enum
import re
from typing import Any, Optional, Type

from occam_core.util.base_models import IOModel
from pydantic import BaseModel, field_validator, model_validator


def remove_extra_spaces(original_text):
    return re.sub(r'\s{2,}', ' ', original_text)


def format_llm_messenger_name(name: str):
    if name:
        return name.replace(" ", "")


class LLMRole(str, enum.Enum):
    assistant = "assistant"
    system = "system"
    user = "user"


class OccamLLMMessage(BaseModel):
    content: str | list[dict[str, Any]]
    role: LLMRole # system vs user vs assistant
    name: Optional[str] = None # this allows us to distinguish users in a multi-user chat
    # Note: this is a pydantic model of the sturcture output if available.
    # at present this means that we can't load this structured output back
    # from the DB as we don't know what the model is.
    # even if we store it as IBaseModel, we have the same issue.
    parsed: Optional[Any] = None

    @model_validator(mode="after")
    def format_messenger_name(cls, v):
        v.name = format_llm_messenger_name(v.name)
        return v


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
