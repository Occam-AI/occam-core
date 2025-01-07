from typing import Any, Optional

from occam_core.llm_utils import LLMInputModel


class AgentIOModel(LLMInputModel):
    extra: Optional[Any] = None
