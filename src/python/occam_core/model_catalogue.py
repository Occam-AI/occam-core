from occam_core.llm_utils import LLMInputModel, LLMOutputModel, LLMParamsModel, OccamLLMMessage
from occam_core.agents.util import AgentIOModel

# TODO: Auto-generate this by scanning the codebase.
MODEL_CATALOGUE = {
    AgentIOModel.__name__: AgentIOModel,
    LLMParamsModel.__name__: LLMParamsModel,
    LLMInputModel.__name__: LLMInputModel,
    LLMOutputModel.__name__: LLMOutputModel,
    OccamLLMMessage.__name__: OccamLLMMessage,
}
