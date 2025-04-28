import enum
import re
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

from occam_core.enums import ToolRunState, ToolState
from occam_core.util.base_models import IOModel
from pydantic import BaseModel, Field, field_validator, model_validator


def remove_extra_spaces(original_text):
    return re.sub(r'\s{2,}', ' ', original_text)


def replace_key_characters(original_text):
    return original_text.\
        replace(' ', '_').\
        replace('.', '_').\
        replace('(', '-').\
        replace(')', '-').\
        replace(':', '--')


def format_llm_messenger_name(name: str):
    if name:
        name = remove_extra_spaces(name)
        name = replace_key_characters(name)
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


# Aliasing tool state as agent state
AgentState = ToolState
AgentRunState = ToolRunState


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

    RESUMED = "RESUMED"
    """Data sent back when the agent announces that it's resumed"""

    STOPPED = "STOPPED"
    """
    Data sent back when the agent announces that it's stopped
    due to a termination request.
    """

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

    USER_SCREEN_MESSAGES = "USER_SCREEN_MESSAGES"
    """User screen data sent back from the agent."""



class TaggedAgentModel(BaseModel):
    tagged_agent_key: str
    tag_type: AgentContactType = AgentContactType.RUN
    tag_message: Optional[str] = None


class TaggedAgentsModel(BaseModel):
    """
    A list of tagged agents.
    """

    tag_models: List[TaggedAgentModel] = Field(default_factory=list)
    tagging_agent_key: str
    _tagged_agent_keys: set[str] = None

    @property
    def tagged_agent_keys(self):
        if not self._tagged_agent_keys:
            self._tagged_agent_keys = set()
        return self._tagged_agent_keys

    @model_validator(mode="after")
    def validate_tag_pairs(self):
        _tagged_agent_keys = set()
        assert len(self.tag_models) > 0, "tag models must not be empty"
        for tag_model in self.tag_models:
            assert self.tagging_agent_key != tag_model.tagged_agent_key, \
                f"tagging agent key must be different from tagged agent key: " \
                f"tagging key: {self.tagging_agent_key} matches tagged key."
            assert tag_model.tagged_agent_key not in _tagged_agent_keys, \
                f"tagged agent key must be unique. Got {tag_model.tagged_agent_key} " \
                f"more than once."
            _tagged_agent_keys.add(tag_model.tagged_agent_key)
        self._tagged_agent_keys = _tagged_agent_keys

        return self

    @classmethod
    def from_keys(cls, tagging_agent_key: str, tagged_agent_keys: set[str] | list[str]):

        tag_models = []
        for key in tagged_agent_keys:
            assert isinstance(key, str), \
                f"tagged agent key must be a string. Got {type(key)}."
            tag_models.append(TaggedAgentModel(tagged_agent_key=key))
        return cls(tagging_agent_key=tagging_agent_key, tag_models=tag_models)

    def append(self, tagged_agent: TaggedAgentModel):
        assert isinstance(tagged_agent, TaggedAgentModel), \
            "tagged agent must be a TaggedAgentModel"
        assert tagged_agent.tagged_agent_key not in self.tagged_agent_keys, \
            "each agent can only be tagged once in a tagged agents list"
        assert tagged_agent.tagged_agent_key != self.tagging_agent_key, \
            "tagged agent key must be different from tagging agent key." \
            f"tagging key: {self.tagging_agent_key} matches tagged key."
        self.tagged_agent_keys.add(tagged_agent.tagged_agent_key)
        self.tag_models.append(tagged_agent)

    def extend(self, tagged_agents: List[TaggedAgentModel]):
        tagged_agent_keys_set = {a.tagged_agent_key for a in tagged_agents}
        assert not (self.tagged_agent_keys & tagged_agent_keys_set), \
            "each agent can only be tagged once in a message."
        self.tagged_agent_keys.update(tagged_agent_keys_set)
        self.tag_models.extend(tagged_agents)
 
    def clear(self):
        self.tag_models.clear()
        self.tagged_agent_keys.clear()
        self.tagging_agent_key = None

    def __len__(self):
        return len(self.tag_models)

    def __getitem__(self, idx):
        return self.tag_models[idx]

    def __iter__(self):
        return iter(self.tag_models)

    def __contains__(self, agent_key: str):
        return agent_key in self.tagged_agent_keys

    def __str__(self):
        return '\n\t'.join([
            f"tagging agent key: {self.tagging_agent_key}",
            f"tagged agent keys: {self.tagged_agent_keys}"
        ])

    def __iadd__(self, other: "TaggedAgentsModel"):
        assert isinstance(other, TaggedAgentsModel), \
            f"other must be a TaggedAgentsModel. Got {type(other)}."
        assert other.tagging_agent_key == self.tagging_agent_key, \
            f"tagging agent key must be the same for all tag models: " \
            f"{self.tagging_agent_key} != {other.tagging_agent_key}"
        self.extend(other.tag_models)
        return self

class ChatStatus(str, enum.Enum):
    # the chat is continuing.
    ACTIVE = "ACTIVE"
    # this is the state while the user is being asked for their chat details.
    CREATING_WORKSPACE = "CREATING_WORKSPACE"

    # the chat is transitioning.
    SPIN_UP_REQUESTED = "SPIN_UP_REQUESTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class OccamLLMMessage(BaseModel):

    type: Literal["base"] = "base"
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
            f"Tagged Agents: \n\t{self.tagged_agents}" if self.tagged_agents else "",
        ]).strip()

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


IOccamLLMMessage = TypeVar("IOccamLLMMessage", bound=OccamLLMMessage)



class ChatManagerOutputMessageModelTemplate(OccamLLMMessage):
    """
    This is to be used at the beginning of the _run of the
    chat manager and to be updated with values as the chat
    manager progresses.
    """

    content: str = ""
    role: LLMRole = LLMRole.assistant

    # this is only expected if the next agent is a human.
    additional_content: Optional[Dict[str, Any]] = None

    # this is to indicate whether the chat manager is sharing a message
    # in the chat or not.
    share_a_message_to_chat: bool = False
    chat_status: ChatStatus = ChatStatus.SUCCESS


class ChatCreatorOutputMessageModel(OccamLLMMessage):
    type: Literal["creator"] = "creator"
    content: str
    role: LLMRole = LLMRole.assistant
    # this is only expected if the next agent is a human.
    additional_content: Optional[Dict[str, Any]] = None
    chat_status: ChatStatus = ChatStatus.CREATING_WORKSPACE


class ChatManagerOutputMessageModel(ChatManagerOutputMessageModelTemplate):

    # these fields are optional in the template but
    # are required to be set in the output.
    # at the beginning of the _run, we initialize them
    # with mock values till they're updated through the
    # run.
    type: Literal["manager"] = "manager"

    chat_status: ChatStatus
    share_a_message_to_chat: bool

    @model_validator(mode='after')
    def validate_additional_content(self):

        if self.chat_status != ChatStatus.ACTIVE and self.tagged_agents:
            raise ValueError(f"Chat manager has terminated the chat but specified a next agent: {self.tagged_agents}.")
        elif self.chat_status != ChatStatus.ACTIVE and self.additional_content:
            raise ValueError("Chat manager has terminated the chat but specified agent selections.")
        if self.share_a_message_to_chat and not self.content:
            raise ValueError("Chat manager has specified to share a message to the chat but has not provided a message.")
        return self


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
    chat_messages: Union[
        None,
        IOccamLLMMessage,
        ChatManagerOutputMessageModel,
        ChatCreatorOutputMessageModel
    ] = None

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
