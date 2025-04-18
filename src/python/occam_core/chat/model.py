



import enum
from datetime import UTC, datetime
from typing import Optional

from occam_core.agents.util import OccamLLMMessage
from pydantic import Field


class ChatPermissions(enum.Enum):
    WRITE = "write"
    END_CHAT = "end_chat"


class ChatStatus(str, enum.Enum):
    # the chat is continuing.
    ACTIVE = "ACTIVE"
    # this is the state while the user is being asked for their chat details.
    CREATING_WORKSPACE = "CREATING_WORKSPACE"

    # the chat is transitioning.
    SPIN_UP_REQUESTED = "SPIN_UP_REQUESTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class MultiAgentWorkspaceCoreMessageModel(OccamLLMMessage):
    """
    This unifies message models of agent members
    and chat managers, so that they can be centralized
    for the front-end in the same DB.
    """

    # instance is run specific
    instance_id: str
    # session can span across independent multi-agent chats
    # runs that we throw in the same place.
    session_id: str

    # Key of the agent that sent the message
    # this is only passed for users, not the chat
    # manager.
    agent_key: Optional[str] = None

    # chat status can occur for human agents and chat managers.
    chat_status: ChatStatus = ChatStatus.ACTIVE

    message_index: Optional[int] = None
    message_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # includes content, plus any extra structured data added to the message.
    full_content: Optional[str] = None
