from enum import Enum


class ToolRunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TERMINATED = "TERMINATED"


# Aliasing tool run status as agent run status
AgentRunStatus = ToolRunStatus
