



import enum
from datetime import UTC, datetime
from typing import Optional

from pydantic import Field

from python.occam_core.agents.util import OccamLLMMessage


class ChatStatus(str, enum.Enum):
    # the chat is continuing.
    ACTIVE = "ACTIVE"
    # this is the state while the user is being asked for their workspace details.
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
    workspace_instance_id: str
    # session can span across independent multi-agent workspaces
    # runs that we throw in the same place.
    workspace_session_id: str

    # Key of the agent that sent the message
    # this is only passed for users, not the chat
    # manager.
    agent_key: Optional[str] = None

    # chat status can occur for human agents and chat managers.
    chat_status: ChatStatus = ChatStatus.ACTIVE

    message_index: Optional[int] = None
    message_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
