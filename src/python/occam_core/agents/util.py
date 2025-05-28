import enum
import re
from datetime import timedelta
from typing import (Any, Dict, List, Literal, Optional, Self, Type, TypeVar,
                    Union)

from occam_core.enums import ToolRunState, ToolState
from occam_core.util.base_models import IOModel
from occam_core.util.data_types.occam import OccamDataType
from pydantic import (BaseModel, ConfigDict, Field, field_serializer,
                      field_validator, model_validator)


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

    INTERMEDIATE_RUN_UPDATES = "INTERMEDIATE_RUN_UPDATES"
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
    tagging_required: bool = False

    tag_message: Optional[str] = None
    tagging_agent_key: Optional[str] = None

    def set_tagging_agent_key(self, tagging_agent_key: str):
        self.tagging_agent_key = tagging_agent_key
        return self


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
            tag_model.set_tagging_agent_key(self.tagging_agent_key)
        self._tagged_agent_keys = _tagged_agent_keys
        return self

    @classmethod
    def from_keys(cls, tagging_agent_key: str, tagged_agent_keys: set[str] | list[str]):

        tag_models = []
        for key in tagged_agent_keys:
            assert isinstance(key, str), \
                f"tagged agent key must be a string. Got {type(key)}."
            tag_models.append(
                TaggedAgentModel(
                    tagged_agent_key=key,
                    tagging_agent_key=tagging_agent_key
                )
            )
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

    # The are workspace templates.
    TEMPLATE = "TEMPLATE"

    #When a workspace is initiated
    # but not run yet.
    STANDBY = "STANDBY"

    # active workspaces (started or resumed).
    ACTIVE = "ACTIVE"

    # when a user terminates a workspace.
    ARCHIVING_REQUESTED = "ARCHIVING_REQUESTED"
    ARCHIVED = "ARCHIVED"

    # when we pause a workspace from
    # the server for whatever reason.
    PAUSED = "PAUSED"


class CallToAction(str, enum.Enum):
    SEND = "Send"
    SAVE = "Save"


class BaseAttachmentModel(BaseModel):
    content: Optional[str | bytes] = None
    cta: Optional[CallToAction] = None
    confirmed: Optional[bool] = None
    name: str

    @model_validator(mode="after")
    def validate_confirmed(self):
        assert not (self.cta is not None and self.confirmed is not None), \
            "attachment cannot have both cta and confirmed set, " \
            "since we either ask for an action or confirm it."
        return self


class EmailAttachmentModel(BaseAttachmentModel):

    # content
    content: str
    subject: str
    sender: str

    # contacts
    recipients: list[str]
    cc: Optional[list[str]] = None
    bcc: Optional[list[str]] = None


class FileMetadataModel(BaseAttachmentModel):
    url: str
    file_key: str
    dataset_uuid: str
    workspace_id: Optional[str] = None

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def set_name_for_workspace_uploads_and_remove_content(self):
        """
        Workspace uploads have a non-user friendly name
        of the form {workspace_id}__{username}__{filename}
        to minimise clash. This function creates a user
        friendly name for workspace upload files

        by taking the filename from the file_key.
        file_key takes the structure:
        {workspace_id}/{filename} so we extract the filename
        and use that as the name.

        This model is also not meant to be used to pass
        content around, so we make sure to remove it.
        """
        if self.workspace_id:
            self.name = self.file_key.split("/")[-1]
        return self


class ReferenceMetadataModel(FileMetadataModel):
    ...


class MessageAttachmentModel(FileMetadataModel):
    """
    We don't dump content of message attachments, as it
    can be loaded back when needed from the database.
    """

    content_type: Optional[str] = None

    @field_serializer('content')
    def serialize_content(self, v, _info):
        if self.content:
            return None
        return v


IAttachmentModel = TypeVar("IAttachmentModel", bound=BaseAttachmentModel)


class MessageType(str, enum.Enum):
    BASE = "base"
    ATTACHMENT = "attachment"
    CREATOR = "creator"
    MANAGER = "manager"


# class StructuredRequestsModel(BaseModel):
#     ...


# IStructuredRequestsModel = TypeVar("IStructuredRequestModel", bound=StructuredRequestsModel)

class StreamMessageType(str, enum.Enum):
    ACTION = "ACTION"
    WARNING = "WARNING"
    SEARCH = "SEARCH"
    ERROR = "ERROR"
    INFO = "INFO"


class StreamingMessageModel(BaseModel):
    message: str
    message_type: StreamMessageType = Field(default=StreamMessageType.ACTION)
    run_time: Optional[timedelta] = None


class StreamingMessagesModel(BaseModel):
    messages: List[StreamingMessageModel] = Field(default_factory=list)

    def append(self, message: StreamingMessageModel):
        self.messages.append(message)

    def extend(self, messages: List[StreamingMessageModel]):
        self.messages.extend(messages)


class OccamLLMMessage(OccamDataType):

    type: MessageType = MessageType.BASE
    """
    This is the type of the message. Used for model validators
    to know how to distinguish between a union of types.
    """

    content: Optional[str | list[dict[str, Any]]] = None
    """
    This is the content of the message.
    """

    content_producer_run_time: Optional[timedelta] = None
    """
    The time it took for the content producer to produce the content.
    """

    update_messages: Optional[StreamingMessagesModel] = None
    """
    A list of update messages. these are sometimes attached
    to a message, and allow us to display "steps or thoughts"
    that may have been involved in generation of the main
    message
    """

    # structured_requests_content: Optional[IStructuredRequestsModel | Dict[str, Any]] = None
    # """
    # This covers things like multiple choice questions, connection
    # wizards, etc.
    # """

    source_attachment: Optional[
        Union[MessageAttachmentModel, EmailAttachmentModel]
    ] = None
    """
    This is the attachment that was used to generate
    the message, if any exists.

    Reason we have this, is that for each attachment, we
    explode it (in case of images usually) to feed into
    LLMs, so we just move it around as a fully fledged
    occam llm message.
    """

    role: LLMRole # system vs user vs assistant
    name: Optional[str] = None

    parsed: Optional[Any] = None
    """
    Note: this is a pydantic model of the sturcture output if available.
    this means that it can't be loaded back from a dump, unless the consumer knows
    what the source model is.
    """

    tagged_agents: Optional[TaggedAgentsModel] = None
    """Agents can tag each other in a message."""

    attachments: Optional[list[Union[MessageAttachmentModel, EmailAttachmentModel]]] = None
    """Attachments are files that can be attached to a message or email attachments.
    We need explicit union here, otherwise pydantic would fail to load them correctly
    when LLM tools are preparing their input (converting AgentIOModel to LLMIOModel)
    """

    content_from_attachments: Optional[list['OccamLLMMessage']] = None
    """Content messages are messages extracted from attachments."""

    references: Optional[list[ReferenceMetadataModel]] = None
    """References to attachments used to generate the message."""

    @field_serializer('content')
    def serialize_content(self, v, info):
        if self.type == MessageType.ATTACHMENT.value and not getattr(info.context, 'keep_content', False):
            return None
        return v

    # @field_serializer('structured_requests_content')
    # def serialize_structured_content(self, v, info):
    #     if v is None:
    #         return None
    #     if isinstance(v, dict):
    #         return v
    #     return v.model_dump(mode="json")

    @model_validator(mode="after")
    def validate_messages(self):
        if self.type == MessageType.ATTACHMENT.value:
            if self.attachments:
                raise ValueError("Messages that represent a single attachment are not expected to have other attachments.")
            if not self.source_attachment:
                raise ValueError("Messages that represent a single attachment must have a source attachment.")
            if not self.content:
                self.content = self.source_attachment.content
        elif self.attachments and not self.content_from_attachments:
            self.content_from_attachments = []
            for attachment in self.attachments:
                self.content_from_attachments.append(OccamLLMMessage.from_attachment(self.role, attachment))

        self.name = format_llm_messenger_name(self.name)
        return self

    def set_attachments(self, attachments: list[MessageAttachmentModel]):
        self.attachments = attachments
        self.content_from_attachments = [OccamLLMMessage.from_attachment(self.role, attachment) for attachment in attachments]

    def set_references(self, references: list[ReferenceMetadataModel]):
        self.references = references

    def set_content_producer_run_time(self, content_producer_run_time: timedelta):
        if self.content_producer_run_time:
            return
        self.content_producer_run_time = content_producer_run_time

    def set_update_messages(self, update_messages: StreamingMessagesModel):
        if self.update_messages and len(self.update_messages.messages) > 0:
            return
        self.update_messages = update_messages

    def to_str(self, message_index: int | None = None):
        return "\n".join([
            f"Message Index: {message_index}" if message_index is not None else "",
            f"Messenger: {self.name}",
            f"Role: {self.role}",
            f"Content: {self.content}",
            f"Tagged Agents: \n\t{self.tagged_agents}" if self.tagged_agents else "",
        ]).strip()

    @classmethod
    def from_attachment(cls, role: LLMRole, attachment: IAttachmentModel) -> Self:
        return cls(
            type=MessageType.ATTACHMENT.value,
            content=attachment.content,
            role=role,
            name=attachment.name,
            source_attachment=attachment
        )

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
    chat_messages: Optional[List[IOccamLLMMessage]] = None

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

    # we dont' save as TypeVar IBaseModel as it can't be
    # loaded back and beaks validation as a result.
    response_format: Optional[Any] = None

    @field_validator('query', 'prompt', mode="before")
    @classmethod
    def transform(cls, raw: str) -> tuple[int, int]:
        if isinstance(raw, str):
            raw = remove_extra_spaces(raw)
        return raw
