from occam_core.agents.params import (CommunicatorAgentParamsModel,
                                      DefinedLLMAgentParamsModel,
                                      LLMAgentParamsModel,
                                      OccamProvidedHumanAgentParamsModel,
                                      UserProvidedHumanAgentParamsModel)
from occam_core.util.base_models import IOModel

# TODO: Auto-generate this by scanning the codebase.
PARAMS_MODEL_CATALOGUE = {
    LLMAgentParamsModel.__name__: LLMAgentParamsModel,
    DefinedLLMAgentParamsModel.__name__: DefinedLLMAgentParamsModel,
    CommunicatorAgentParamsModel.__name__: CommunicatorAgentParamsModel,
    UserProvidedHumanAgentParamsModel.__name__: UserProvidedHumanAgentParamsModel,
    OccamProvidedHumanAgentParamsModel.__name__: OccamProvidedHumanAgentParamsModel,
}

IO_MODEL_CATALOGUE = {
    IOModel.__name__: IOModel,
}
