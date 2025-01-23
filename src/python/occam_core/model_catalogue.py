from occam_core.agents.params import (AgentsChatParamsModel,
                                      DataStructuringAgentParamsModel,
                                      DefinedLLMAgentParamsModel,
                                      EmailCommunicatorAgentParamsModel,
                                      LLMAgentParamsModel,
                                      OccamProvidedHumanAgentParamsModel,
                                      UserProvidedHumanAgentParamsModel)
from occam_core.util.base_models import IOModel

# TODO: Auto-generate this by scanning the codebase.
PARAMS_MODEL_CATALOGUE = {
    AgentsChatParamsModel.__name__: AgentsChatParamsModel,
    DataStructuringAgentParamsModel.__name__: DataStructuringAgentParamsModel,
    EmailCommunicatorAgentParamsModel.__name__: EmailCommunicatorAgentParamsModel,
    LLMAgentParamsModel.__name__: LLMAgentParamsModel,
    UserProvidedHumanAgentParamsModel.__name__: UserProvidedHumanAgentParamsModel,
    OccamProvidedHumanAgentParamsModel.__name__: OccamProvidedHumanAgentParamsModel,
    DefinedLLMAgentParamsModel.__name__: DefinedLLMAgentParamsModel,
}

IO_MODEL_CATALOGUE = {
    IOModel.__name__: IOModel,
}
