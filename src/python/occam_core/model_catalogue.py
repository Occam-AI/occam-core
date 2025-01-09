from occam_core.agents.params import (EmailCommunicatorParamsModel,
                                      LLMParamsModel, UserAgentParamsModel)
from occam_core.util.base_models import IOModel, ParamsIOModel

# TODO: Auto-generate this by scanning the codebase.
PARAMS_MODEL_CATALOGUE = {
    ParamsIOModel.__name__: ParamsIOModel,
    LLMParamsModel.__name__: LLMParamsModel,
    EmailCommunicatorParamsModel.__name__: EmailCommunicatorParamsModel,
    UserAgentParamsModel.__name__: UserAgentParamsModel,
}

IO_MODEL_CATALOGUE = {
    IOModel.__name__: IOModel,
}
