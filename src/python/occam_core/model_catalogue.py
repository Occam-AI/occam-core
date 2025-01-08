from occam_core.agents.model import AgentIOModel
from occam_core.agents.util import LLMInputModel, OccamLLMMessage
from occam_core.util.base_models import IOModel, ParamsIOModel

# TODO: Auto-generate this by scanning the codebase.
MODEL_CATALOGUE = {
    AgentIOModel.__name__: AgentIOModel,
    IOModel.__name__: IOModel,
    ParamsIOModel.__name__: ParamsIOModel,
}
