from occam_core.agents.params import (EmailCommunicatorParamsModel,
                                      LLMParamsModel,
                                      UserAgentParticipationParamsModel)
from occam_core.util.base_models import IOModel, ParamsIOModel

# TODO: Auto-generate this by scanning the codebase.
PARAMS_MODEL_CATALOGUE = {
    LLMParamsModel.__name__: LLMParamsModel,
    EmailCommunicatorParamsModel.__name__: EmailCommunicatorParamsModel,
    UserAgentParticipationParamsModel.__name__: UserAgentParticipationParamsModel,
}

IO_MODEL_CATALOGUE = {
    IOModel.__name__: IOModel,
}
