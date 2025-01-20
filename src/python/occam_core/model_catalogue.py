from occam_core.agents.params import (CommunicatorAgentParamsModel,
                                      LLMParamsModel,
                                      OccamInterfaceAgentPermissionsModel)
from occam_core.util.base_models import IOModel, ParamsIOModel

# TODO: Auto-generate this by scanning the codebase.
PARAMS_MODEL_CATALOGUE = {
    LLMParamsModel.__name__: LLMParamsModel,
    CommunicatorAgentParamsModel.__name__: CommunicatorAgentParamsModel,
    OccamInterfaceAgentPermissionsModel.__name__: OccamInterfaceAgentPermissionsModel,
}

IO_MODEL_CATALOGUE = {
    IOModel.__name__: IOModel,
}
