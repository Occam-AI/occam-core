import enum
import hashlib
import re
from datetime import datetime, timedelta
from typing import (Any, Dict, List, Literal, Optional, Self, Type, TypeVar,
                    Union)

from occam_core.enums import ToolRunState, ToolState
from occam_core.util.base_models import IOModel
from occam_core.util.data_types.occam import OccamDataType
from openai.types.chat import ParsedChatCompletionMessage
from openai.types.chat.chat_completion import ChoiceLogprobs, CompletionUsage
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

    I_AM_ALIVE = "I_AM_ALIVE"
    """
    This is a special message that is routinely sent by
    the agent listener to declare that it's alive
    """


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
    content_type: Optional[str] = None
    cta: Optional[CallToAction] = None
    name: str
    display_name: str = Field(default_factory=str)
    """
    This is the name of the file as it appears in the
    workspace.
    """
    attachment_id: Optional[str] = None
    """
    This is the id of the attachment in the workspace.
    This will always be set by the model validator
    on child classes.
    """
    workspace_id: Optional[str] = None
    

    @model_validator(mode="after")
    def ensure_size_limit_for_name_and_add_display_name(self):
        """
        Workspace uploads have a non-user friendly name
        of the form {workspace_id}__{user_email}__{filename}
        to minimise clash. This function creates a user
        friendly name for workspace upload files.

        We also ensure that the attachment name is under 64 chars,
        as otherwise it would lead to issues with LLM when
        attachments are exploded into messages.
        """
        self.display_name = self.name
        if self.workspace_id:
            parts = self.name.split("__")
            self.display_name = parts[-1]
            self.name = self.display_name[:64]
        return self


class FileMetadataModel(BaseAttachmentModel):
    url: str
    file_key: str
    datasource_uuid: str
    dataset_uuid: str
    size_kb: Optional[int] = None

    model_config = ConfigDict(extra="ignore")

class ReferenceMetadataModel(FileMetadataModel):
    ...


class FileAttachmentModel(FileMetadataModel):
    """
    We don't dump content of message attachments, as it
    can be loaded back when needed from the database.
    """

    @field_serializer('content')
    def serialize_content(self, v, _info):
        if self.content:
            return None
        return v


class MailContactModel(BaseModel):
    full_name: str
    primary_email: str
    email_addresses: Optional[List[str]] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    address: Optional[str] = None


class EmailDraftProposalResponseEnum(str, enum.Enum):
    SEND = "SEND"
    REDRAFT = "RE-DRAFT"
    DISCARD = "DISCARD"


class EmailAttachmentMetadataModel(BaseModel):
    """
    This model serves two primary functions:

    1. As a base for the email attachment model that's

    A. sent to the front-end to display a proposal to the user of a draft email.

    B. Sent to the front-end to display messages fetched from a user's
    inbox.

    2. As part of the user send approval model, to trace what a user
       approved to send.

    Specifically, when it comes to file attachments, they also serve 2 purposes:

    1. When they're part of the sent draft, it means a proposal for
       pre-uploaded files, whether generated by the inbox agent, or an
       explicit user upload prior to draft generation.

    2. When they're part of the send approval, it captures
    the full set of attachments that would be included in the actual
    email being sent, whether these came in the proposal, or were
    added/removed by the user.
    """

    # Reads.
    sent_date: Optional[datetime] = None
    """
    Optional because we only get this for emails we
    retrieve through fetching emails from the user's
    inbox, so this is not available for emails we draft
    ourselves.
    """

    # Send Approvals.
    draft_proposal_response: Optional[EmailDraftProposalResponseEnum] = None

    # Reads to web-app, drafts to webapp, send approvals to back-end.
    subject: str
    sender: MailContactModel
    snippet: str
    recipients: list[MailContactModel]
    cc: Optional[list[MailContactModel]] = None
    bcc: Optional[list[MailContactModel]] = None
    # attachment is is required when this model is loaded
    # by the front-end directly to confirm a user send.
    attachment_id: Optional[str] = None
    # This is needed for sending draft edits.
    content: Optional[str] = None

    file_attachments: Optional[list[FileAttachmentModel | BaseAttachmentModel]] = None
    """
    BaseAttachmentModel is supported here as a way to get attachments to emails
    from the user inbox before we create datasets for them and have the other
    required fields in FileAttachmentModel. This is all internal to the inbox agent
    though and base attachments are converted to FileAttachmentModel before results
    are returned from the agent.
    """

    @model_validator(mode="after")
    def validate_attachment_id_required(self):
        # Only validate if this is a direct instantiation of 
        # EmailAttachmentMetadataModel, not a subclass
        if (type(self) is EmailAttachmentMetadataModel and 
            self.attachment_id is None):
            raise ValueError(
                "attachment_id is required when EmailAttachmentMetadataModel "
                "is directly instantiated"
            )
        return self


class EmailAttachmentModel(BaseAttachmentModel, EmailAttachmentMetadataModel):

    content: str

    @model_validator(mode="after")
    def set_attachment_id(self):

        self.attachment_id = hashlib.sha256(
            "".join([
                self.content,
                self.subject,
                self.sender.primary_email,
                ",".join([r.primary_email for r in self.recipients]),
                ",".join([r.primary_email for r in (self.cc or [])]),
                ",".join([r.primary_email for r in (self.bcc or [])])
            ]).encode('utf-8')
        ).hexdigest()

        return self


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


class MessageLabelEnum(str, enum.Enum):

    WARNING = "WARNING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    GENERAL = "GENERAL"


class MessageEvaluationModel(BaseModel):
    label: MessageLabelEnum
    message: str

    @model_validator(mode="after")
    def validate_message(self):
        assert len(self.message) <= 100, "message must be no longer than 100 characters"
        return self


class OccamLLMMessage(OccamDataType, ParsedChatCompletionMessage):

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

    action_evaluation: Optional[MessageEvaluationModel] = None
    """
    This is used when we want to append an evaluation to a
    message, in case where it relates to a task, that may have
    failed, or succeeded etc.
    """

    # structured_requests_content: Optional[IStructuredRequestsModel | Dict[str, Any]] = None
    # """
    # This covers things like multiple choice questions, connection
    # wizards, etc.
    # """

    source_attachment: Optional[
        Union[FileAttachmentModel, EmailAttachmentModel]
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
    Note: this is a pydantic model of the sturcture output
    if available. this means that it can't be loaded back
    from a dump, unless the consumer knows what the source
    model is.
    """

    tagging_agent_keys_chain: Optional[List[str]] = None
    """
    The chain of agents that ran on the way to us
    triggering this agent.
    """

    agent_key: Optional[str] = None
    """
    The key of the agent that generated the message,
    if applicable.
    """

    tagged_agents: Optional[TaggedAgentsModel] = None
    """
    Agents that were tagged by this message and
    sent this message.
    """

    attachments: Optional[list[Union[FileAttachmentModel, EmailAttachmentModel]]] = None
    """
    Attachments are files that can be attached to a message
    or email attachments. We need explicit union here,
    otherwise pydantic would fail to load them correctly
    when LLM tools are preparing their input
    (converting AgentIOModel to LLMIOModel)
    """

    content_from_attachments: Optional[list['OccamLLMMessage']] = None
    """Content messages are messages extracted from attachments."""

    references: Optional[list[ReferenceMetadataModel]] = None
    """References to attachments used to generate the message."""


    # these come from standard LLM calls.
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
    """The reason the model stopped generating tokens.

    This will be `stop` if the model hit a natural stop point or a provided stop
    sequence, `length` if the maximum number of tokens specified in the request was
    reached, `content_filter` if content was omitted due to a flag from our content
    filters, `tool_calls` if the model called a tool, or `function_call`
    (deprecated) if the model called a function.
    """

    index: Optional[int] = None
    """The index of the choice in the list of choices."""

    logprobs: Optional[ChoiceLogprobs] = None
    """Log probability information for the choice."""


    @field_serializer('content')
    def serialize_content(self, v, info):
        if self.type == MessageType.ATTACHMENT.value and not getattr(info.context, 'keep_content', False):
            return None
        return v

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

    def set_attachments(self, attachments: list[IAttachmentModel]):
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
            # FIXME FIXME FIXME. OpenAI fails if role is assistant for image attachments.
            # Temporarily setting it to user.
            role=LLMRole.user,
            # Definisevly guard against long names (max 64 for OpenAI)
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

    # these fields come back from ChatCompletion responses of LLMs.
    id: Optional[str] = None
    """A unique identifier for the chat completion."""

    model: Optional[str] = None
    """The model used for the chat completion."""

    service_tier: Optional[Literal["scale", "default"]] = None
    """The service tier used for processing the request."""

    system_fingerprint: Optional[str] = None
    """This fingerprint represents the backend configuration that the model runs with.

    Can be used in conjunction with the `seed` request parameter to understand when
    backend changes have been made that might impact determinism.
    """

    usage: Optional[CompletionUsage] = None
    """Usage statistics for the completion request."""

    @field_validator('query', 'prompt', mode="before")
    @classmethod
    def transform(cls, raw: str) -> tuple[int, int]:
        if isinstance(raw, str):
            raw = remove_extra_spaces(raw)
        return raw
