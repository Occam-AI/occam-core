from occam_core.agents.util import LLMParamsModel
from occam_core.util.base_models import IOModel, ParamsIOModel

# TODO: Auto-generate this by scanning the codebase.
PARAMS_MODEL_CATALOGUE = {
    ParamsIOModel.__name__: ParamsIOModel,
    LLMParamsModel.__name__: LLMParamsModel,
}

IO_MODEL_CATALOGUE = {
    IOModel.__name__: IOModel,
}
