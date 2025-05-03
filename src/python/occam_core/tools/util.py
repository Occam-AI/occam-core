import enum
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from occam_core.chat.model import ChatPermissions
from occam_core.enums import ToolRunState, ToolState
from pydantic import BaseModel, Field, model_validator


class OccamUUID(uuid.UUID):
    """
    A subclass of uuid.UUID that ensures the UUID is a valid Occam UUID.
    """

    @staticmethod
    def uuid4_no_dash():
        return str(uuid.uuid4()).replace("-", "_")



class ToolInstanceType(str, enum.Enum):
    TOOL = "TOOL"
    AGENT = "AGENT"
    WORKSPACE = "WORKSPACE"


class ToolInstanceContext(BaseModel):

    instance_id: Optional[str] = None
    """
    The id of the instance for checkpointing and tracking.
    """

    instance_type: Optional[ToolInstanceType] = None
    """
    The type of the instance, whether it's a tool, agent or workspace.
    """

    workspace_id: Optional[str] = None
    """
    The id of the workspace that the tool is associated with.
    Tool has to be instantiated as an agent in a workspace.
    """

    workspace_permissions: Optional[List[ChatPermissions]] = None
    """
    The permissions for the workspace, whether an
    agent can write, end the workspace etc.
    """

    session_id: Optional[str] = None
    """
    The session id for the tool. Session id signifies
    an incubator that the tool's work into. This can
    span across many different runs and tools. As such
    is a more general concept than an instance id.
    """

    agent_key: Optional[str] = None
    """
    The key for the agent that is controlling the tool.
    """

    instance_link: Optional[str] = None
    """
    The base link for the tool instance.
    """

    run_link: Optional[str] = None
    """
    The link for tracking the tool as it works.
    """

    state: Optional[ToolState] = None
    """
    The state of the tool.
    """

    run_state: Optional[ToolRunState] = None
    """
    The state of the tool run.
    """



    # this is to allow the tool to control the
    # state of the agent if allowed to do so.
    allow_external_state_control: bool = False
    """
    Whether the tool is allowed to control the state of the agent.
    """

    last_sent_times_by_sender_id: Optional[Dict[str, datetime]] \
        = Field(default_factory=dict)
    """
    A log of the last times that the tool instance has
    received a message, indexed on the instance id of the
    sender of the message. Use for checkpointing communication
    with other tools. i.e. not having memory loss.
    """

    last_channel_sent_time: Optional[datetime] = None
    """
    The last time that the tool instance has received a message
    through any channel.
    """

    extra: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @model_validator(mode="after")
    def validators(self):
        self.set_ids()
        self.check_instance_type()
        self.check_workspace_permissions()
        return self

    def set_ids(self):
        if self.instance_id is None:
            self.instance_id = OccamUUID.uuid4_no_dash()
        if self.session_id is None:
            self.session_id = OccamUUID.uuid4_no_dash()

    def check_instance_type(self):
        if self.workspace_id is not None:
            assert self.agent_key is not None, \
                "agent key must be set if workspace id is set"
            assert self.instance_type in [ToolInstanceType.AGENT, None], \
                "instance type must be set to AGENT or None if workspace id is set"
            self.instance_type = ToolInstanceType.AGENT
        elif self.agent_key is not None:
            assert self.instance_type in [None, ToolInstanceType.WORKSPACE, ToolInstanceType.AGENT], \
                "instance type must be set to TOOL if agent key is set"
            # knowledge of instance type being a workspace lands externally (relies on agent key)
            # so here we only label it as agent if not speified.
            self.instance_type = self.instance_type or ToolInstanceType.AGENT
        else:
            assert self.instance_type in [None, ToolInstanceType.TOOL], \
                "instance type must be set to TOOL if agent key is not set"
            self.instance_type = ToolInstanceType.TOOL

    def check_workspace_permissions(self):
        if self.workspace_id is not None and not self.workspace_permissions:
            self.workspace_permissions = [ChatPermissions.WRITE]

    def update_run_link(self, run_link: str):
        self.run_link = run_link
    
    def update_instance_link(self, instance_link: str):
        self.instance_link = instance_link
