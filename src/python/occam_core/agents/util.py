import enum
import re
from typing import Any, List, Optional, Type, TypeVar

from occam_core.enums import ToolRunStatus
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



class AgentOutputType(str, Enum):
    ALL = "ALL"
    LATEST = "LATEST"
    NONE = "NONE"


# Aliasing tool run status as agent run status
AgentStatus = ToolRunStatus


class AgentContactType(str, enum.Enum):
    STATUS = "STATUS"
    RUN = "RUN"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    STOP = "STOP"
    OUTPUT = "OUTPUT"
    GENERAL = "GENERAL"
    ERROR = "ERROR"
    OTHER = "OTHER"


class TaggedAgentModel(BaseModel):
    agent_key: str
    tag_type: Optional[AgentContactType] = None
    tag_message: Optional[str] = None


class TaggedAgentsModel(BaseModel):
    tagged_agents: List[TaggedAgentModel] = Field(default_factory=list)
    _agent_keys: set[str] = Field(default_factory=set)

    def append(self, tagged_agent: TaggedAgentModel):
        self.tagged_agents.append(tagged_agent)
        assert tagged_agent.agent_key not in self._agent_keys, \
            "each agent can only be tagged once in a tagged agents list"
        self._agent_keys.add(tagged_agent.agent_key)

    def extend(self, tagged_agents: List[TaggedAgentModel]):
        self.tagged_agents.extend(tagged_agents)
        tagged_agent_keys_set = {a.agent_key for a in tagged_agents}
        assert self._agent_keys - tagged_agent_keys_set == set(), \
            "each agent can only be tagged once in a message"
        self._agent_keys.update(tagged_agent_keys_set)

    def clear(self):
        self.tagged_agents.clear()
        self._agent_keys.clear()

    def __len__(self):
        return len(self.tagged_agents)

    def __getitem__(self, idx):
        return self.tagged_agents[idx]

    def __iter__(self):
        return iter(self.tagged_agents)

    def __contains__(self, agent_key: str):
        return agent_key in self._agent_keys

    def __str__(self):
        return str(self._agent_keys)


class OccamLLMMessage(BaseModel):
    content: str | list[dict[str, Any]]
    role: LLMRole # system vs user vs assistant
    name: Optional[str] = None # this allows us to distinguish users in a multi-user chat
    # Note: this is a pydantic model of the sturcture output if available.
    # at present this means that we can't load this structured output back
    # from the DB as we don't know what the model is.
    # even if we store it as IBaseModel, we have the same issue.
    parsed: Optional[Any] = None

    tagged_agents: Optional[TaggedAgentsModel] = None

    @model_validator(mode="after")
    def validate_messages(self):
        self.name = format_llm_messenger_name(self.name)
        return self

    def to_str(self, message_index: int | None = None):
        return "\n".join([
            f"Message Index: {message_index}" if message_index is not None else "",
            f"Messenger: {self.name}",
            f"Role: {self.role}",
            f"Content: {self.content}",
            f"Tagged Agents: {self.tagged_agents}" if self.tagged_agents else "",
        ]).strip()

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

    # name of the user that's sending the llm model.
    # usually this is in case where we're using top-level
    # prompt query and role, if instead we're relying on chat_messages
    # each message has its own quadruple.
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
