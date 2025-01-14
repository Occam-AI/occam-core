import inspect
from enum import Enum
from inspect import isabstract
from typing import Any, Dict, List, Optional, Type

from occam_core.agents.util import LLMInputModel
from occam_core.util.base_models import IOModel
from pydantic import BaseModel, model_validator


class AgentType(str, Enum):
    OCCAM_AGENT = "OCCAM_AGENT"
    USER = "USER"
    EXTERNAL_WORKER = "EXTERNAL_WORKER"


class AgentRole(str, Enum):
    GENERAL = "general"
    LANGUAGE_MODEL = "language_model"
    todo = "todo"


class AgentIdentityCoreModel(BaseModel):
    """
    This model enforces field requirements based on AgentType.
    """

    name: str
    # for humans
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    # the tool that's used to run the agent.
    base_agent: str
    type: AgentType
    role: AgentRole
    role_embedding_vector: Optional[List[float]] = None
    short_role_description: Optional[str] = None
    long_role_description: Optional[str] = None
    email: Optional[str] = None
    slack_handle: Optional[str] = None

    # for agents that are users.
    user_id: Optional[int] = None
    # for agents that aren't users.
    prompt: Optional[str] = None

    # for the user to be able to specify the params model
    # to instantiate the agent with.
    params_model_name: str

    # the agent defining params and dynamic
    # spec of the tool.
    # if neither are passed, then this agent is
    # simply the agentic tool itself, regardless
    # of which parameters it's run with.
    partial_params: Optional[IOModel] = None
    dynamic_spec: Optional[Dict[str, Any]] = None
    is_base_agent: bool = False

    @model_validator(mode="after")
    def validate_agent_fields(self):
        base_agent_kind = self.base_agent
        dynamic_spec = self.dynamic_spec
        agent_type = self.type
        user_id = self.user_id
        prompt = self.prompt
        email = self.email
        first_name = self.first_name
        last_name = self.last_name

        if self.is_base_agent:
            if self.partial_params is not None or self.dynamic_spec is not None:
                raise ValueError("base agent must have no params or dynamic spec")
            if (
                inspect.isclass(base_agent_kind) and self.name != base_agent_kind.__name__
            ) or (
                isinstance(base_agent_kind, str) and self.name != base_agent_kind
            ):
                raise ValueError(f"agent {self.name}'s base agent must have name matching base tool kind.")
        
        # FIXME: revert once occam-tools stable and all models moved here.
        # if self.params_model_name not in PARAMS_MODEL_CATALOGUE:
        #     raise ValueError(f"agent {self.name}'s params model {self.params_model_name} not found in params_model catalogue.")

        if agent_type == AgentType.USER:
            if first_name is None or last_name is None:
                raise ValueError(f"agent {self.name}'s first_name and last_name are required for {AgentType.USER} agents.")
            if user_id is None:
                raise ValueError(f"agent {self.name}'s user_id missing,required for {AgentType.USER} agents.")
            if prompt is not None:
                raise ValueError(f"prompt must be None for {AgentType.USER} agents.")
            if email is None:
                raise ValueError(f"agent {self.name}'s email is required for {AgentType.USER} agents.")
        elif agent_type == AgentType.EXTERNAL_WORKER:
            if user_id is not None:
                raise ValueError(f"agent {self.name}'s user_id must be None for {AgentType.EXTERNAL_WORKER} agents.")
            if prompt is None:
                raise ValueError(f"agent {self.name}'s prompt is missing,required for {AgentType.EXTERNAL_WORKER} agents.")
            if email is None:
                raise ValueError(f"agent {self.name}'s email is missing, required for {AgentType.EXTERNAL_WORKER} agents.")
        elif agent_type == AgentType.OCCAM_AGENT:
            if user_id is not None:
                raise ValueError(f"agent {self.name}'s user_id must be None for {AgentType.OCCAM_AGENT} agents.")
            if prompt is None:
                raise ValueError(f"agent {self.name}'s prompt missing, required for {AgentType.OCCAM_AGENT} agents.")
        else:
            raise ValueError(f"agent {self.name}'s unknown agent type: {agent_type}")

        return self


class AgentsCatalogueModel(BaseModel):
    agents: Dict[str, AgentIdentityCoreModel]

    @model_validator(mode="after")
    def validate_agents(self):
        for agent_name, agent_model in self.agents.values():
            agent_model: AgentIdentityCoreModel
            if agent_name != agent_model.name:
                raise ValueError(f"agent name {agent_name} doesn't match agent model name {agent_model.name}")
        return self


class AgentIOModel(LLMInputModel):
    extra: Optional[Any] = None
