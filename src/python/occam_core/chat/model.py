



import enum
from datetime import UTC, datetime
from typing import Optional

from occam_core.agents.util import ChatStatus, OccamLLMMessage
from pydantic import Field


class ChatPermissions(enum.Enum):
    WRITE = "write"
    END_CHAT = "end_chat"


class MultiAgentWorkspaceCoreMessageModel(OccamLLMMessage):
    """
    This unifies message models of agent members
    and chat managers, so that they can be centralized
    for the front-end in the same DB.
    """

    instance_id: str
    """
    the id of the agent instance that generated the message
    """

    session_id: str = None
    """
    the id of the session in which the message was sent. This
    can span multiple workspaces.
    """

    workspace_id: str
    """
    the id of the workspace in which the message was sent.
    """

    agent_key: Optional[str] = None
    """
    the key of the agent that sent the message.
    """

    chat_status: ChatStatus = ChatStatus.ACTIVE
    """
    the status of the chat designated by the communicating agent
    (if they are provided with the permissions to do so)
    """

    message_index: Optional[int] = None
    """
    the index of the message in the chat.
    """

    message_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    """
    the time the message was sent.
    """

    full_content: Optional[str] = None
    """
    the full text content of the message.
    """
