import enum
import re
from typing import Any, List, Optional, Type, TypeVar

from occam_core.util.base_models import IOModel
from pydantic import BaseModel, Field, field_validator, model_validator


def remove_extra_spaces(original_text):
    return re.sub(r'\s{2,}', ' ', original_text)


def format_llm_messenger_name(name: str):
    if name:
        name = remove_extra_spaces(name)
        pat = r'[^a-zA-Z0-9_-]'
        return re.sub(pat, '', name)


class LLMRole(str, enum.Enum):
    assistant = "assistant"
    system = "system"
    developer = "developer"
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

    tagged_agents: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_messages(self):
        self.name = format_llm_messenger_name(self.name)
        if self.tagged_agents:
            assert (len(self.tagged_agents) == len(set(self.tagged_agents))), "tagged_agents must must be unique."
        return self

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


IOccamLLMMessage = TypeVar("IOccamLLMMessage", bound=OccamLLMMessage)


class LLMIOModel(IOModel):
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
    chat_messages: Optional[list[IOccamLLMMessage]] = None

    # intermediate prompt that can be used to guide interpretation
    # of the message to follow.
    prompt: Optional[str] = None
    # llm expects either a single query or a list of chat messages
    query: Optional[str] = None

    # role of the user providing the query or chat messages
    role: LLMRole = LLMRole.user

    # name of the user, eg. engineer, architect etc.
    name: Optional[str] = None

    # file paths to be used for vision models
    file_paths: Optional[list[str]] = None

    # we dont' save as TypeVar IBaseModel as it can't be
    # loaded back and beaks validation as a result.
    response_format: Optional[Any] = None

    @field_validator('query', 'prompt', mode="before")
    @classmethod
    def transform(cls, raw: str) -> tuple[int, int]:
        if isinstance(raw, str):
            raw = remove_extra_spaces(raw)
        return raw
