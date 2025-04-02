import uuid
from typing import Any, Optional

from pydantic import BaseModel, model_validator


class OccamUUID(uuid.UUID):
    """
    A subclass of uuid.UUID that ensures the UUID is a valid Occam UUID.
    """

    @staticmethod
    def uuid4_no_dash():
        return str(uuid.uuid4()).replace("-", "_")


class ToolInstanceContext(BaseModel):

    # this is track the init instance
    # for checkpointing and tracking.
    instance_id: Optional[str] = OccamUUID.uuid4_no_dash()

    # this is to track the channel
    # in which tool activity ends up
    session_id: Optional[str] = OccamUUID.uuid4_no_dash()

    # ths is in case this is an agentic tool
    # and it's being launched by agent
    # agent_key. otherwise there's no
    # way for us to tell who the agent is.
    agent_key: Optional[str] = None

    # this is the link for tracking the agent as it works.
    run_link: Optional[str] = None

    extra: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @model_validator(mode="after")
    def check_instance_id(self):
        if self.instance_id is None:
            self.instance_id = OccamUUID.uuid4_no_dash()
        if self.session_id is None:
            self.session_id = OccamUUID.uuid4_no_dash()
        return self
