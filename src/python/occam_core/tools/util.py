import enum
import uuid
from typing import Any, List, Optional

from occam_core.chat.model import ChatPermissions
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

    # This is track the init instance
    # for checkpointing and tracking.
    instance_id: Optional[str] = None
    instance_type: ToolInstanceType = None
    workspace_id: Optional[str] = None
    workspace_permissions: Optional[List[ChatPermissions]] = None

    # This is to track the channel
    # in which tool activity ends up
    session_id: Optional[str] = None

    # Ths is in case this is an agentic tool
    # and it's being launched by agent
    # agent_key. otherwise there's no
    # way for us to tell who the agent is.
    agent_key: Optional[str] = None

    # this is the link for tracking the agent as it works.
    run_link: Optional[str] = None

    # this is to allow the tool to control the
    # state of the agent if allowed to do so.
    allow_external_state_control: bool = False

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
