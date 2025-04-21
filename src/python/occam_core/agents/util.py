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



class AgentOutputType(str, enum.Enum):
    ALL = "ALL"
    LATEST = "LATEST"
    NONE = "NONE"


# Aliasing tool run status as agent run status
AgentStatus = ToolRunStatus


class AgentContactType(str, enum.Enum):
    """
    This list defines the different types of messages
    that can agent can receive or send over channels
    to other agents, workspaces and the wider ecosystem.
    """

    STATUS = "STATUS"
    RUN = "RUN"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    STOP = "STOP"
    """These are requests that can be sent to the agent"""

    STREAMING_UPDATES = "STREAMING_UPDATES"
    """Internal updates sent back from the agent sometimes this won't exist."""

    PAUSED = "PAUSED"
    """Data sent back when the agent announces that it's paused"""

    RUN_STEP_COMPLETED = "RUN_STEP_COMPLETED"
    """Data sent back when the agent announces that a batch step has completed."""

    OUTPUT = "OUTPUT"
    """
    Data sent back when the agent announces that it's completed a full streaming/batch run
    call involving a list of input records or an AgentIOModel.
    """

    GENERAL = "GENERAL"
    """General purpose data sent back from the agent."""

    ERROR = "ERROR"
    """Error data sent back from the agent."""

    OTHER = "OTHER"
    """Other data sent back from the agent."""



class TaggedAgentModel(BaseModel):
    agent_key: str
    tag_type: AgentContactType = AgentContactType.GENERAL
    tag_message: Optional[str] = None


class TaggedAgentsModel(BaseModel):
    tagged_agents: List[TaggedAgentModel] = Field(default_factory=list)
    _agent_keys: set[str] = None

    @property
    def agent_keys(self):
        if not self._agent_keys:
            self._agent_keys = set()
        return self._agent_keys

    @classmethod
    def from_keys(cls, agent_keys: set[str] | list[str]):
        return cls(tagged_agents=[TaggedAgentModel(agent_key=key) for key in agent_keys])

    def append(self, tagged_agent: TaggedAgentModel):
        self.tagged_agents.append(tagged_agent)
        assert tagged_agent.agent_key not in self.agent_keys, \
            "each agent can only be tagged once in a tagged agents list"
        self.agent_keys.add(tagged_agent.agent_key)

    def extend(self, tagged_agents: List[TaggedAgentModel]):
        self.tagged_agents.extend(tagged_agents)
        tagged_agent_keys_set = {a.agent_key for a in tagged_agents}
        assert self.agent_keys - tagged_agent_keys_set == set(), \
            "each agent can only be tagged once in a message"
        self.agent_keys.update(tagged_agent_keys_set)

    def clear(self):
        self.tagged_agents.clear()
        self.agent_keys.clear()

    def __len__(self):
        return len(self.tagged_agents)

    def __getitem__(self, idx):
        return self.tagged_agents[idx]

    def __iter__(self):
        return iter(self.tagged_agents)

    def __contains__(self, agent_key: str):
        return agent_key in self.agent_keys

    def __str__(self):
        return str(self.agent_keys)

    def __iadd__(self, other: "TaggedAgentsModel"):
        self.extend(other.tagged_agents)
        return self


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
