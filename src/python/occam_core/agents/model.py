import inspect
from enum import Enum
from inspect import isabstract
from typing import Any, Dict, List, Optional, Type, TypeVar

from occam_core.agents.util import LLMInputModel
from occam_core.model_catalogue import PARAMS_MODEL_CATALOGUE
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

    # name information.
    key: str
    # some agents' name is based on key, for humans, it's first and last name.
    name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None

    # agent type and role.
    type: AgentType
    role: AgentRole

    # agent descriptors
    role_embedding_vector: Optional[List[float]] = None
    short_role_description: Optional[str] = None
    long_role_description: Optional[str] = None

    # agent contact information
    email: Optional[str] = None
    slack_handle: Optional[str] = None

    # for loading the params model
    # needed for instantiating the agent with.
    instance_params_model_name: str

    @model_validator(mode="after")
    def validate_params_model_name(self):
        # NOTE: @Medhat I added this type check because the validator was wrong.
        # It runs when AgentIdentityModels are instantiated (not just AgentIdentityCoreModels),
        # and the param_model_name for those is not (and will not be) in PARAMS_MODEL_CATALOGUE.
        # For now I added this check to skip validation if self is not an AgentIdentityCoreModel.
        # FIXME: Remove this if and skip condition once all params models saved in DB are present
        # in PARAMS_MODEL_CATALOGUE.
        if type(self) != AgentIdentityCoreModel:
            return self
        if self.instance_params_model_name not in PARAMS_MODEL_CATALOGUE:
            raise ValueError(f"agent {self.name}'s params model {self.instance_params_model_name} not found in params_model catalogue.")
        return self


class AgentsCatalogueModel(BaseModel):
    agents: Dict[str, AgentIdentityCoreModel]

    @model_validator(mode="after")
    def validate_agents(self):
        for agent_name, agent_model in self.agents.items():
            agent_model: AgentIdentityCoreModel
            if agent_name != agent_model.name:
                raise ValueError(f"agent name {agent_name} doesn't match agent model name {agent_model.name}")
        return self


class AgentIOModel(LLMInputModel):
    extra: Optional[Any] = None


TAgentIOModel = TypeVar("TAgentIOModel", bound=AgentIOModel)
