from occam_core.llm_utils import LLMInputModel, OccamLLMMessage
from occam_core.agents.util import AgentIOModel

# TODO: Auto-generate this by scanning the codebase.
MODEL_CATALOGUE = {
    AgentIOModel.__name__: AgentIOModel,
    LLMInputModel.__name__: LLMInputModel,
    OccamLLMMessage.__name__: OccamLLMMessage,
}
