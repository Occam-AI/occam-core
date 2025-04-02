from enum import Enum


class ToolRunStatus(str, Enum):
    #These are states where the instance exists
    # but no run is active.
    # alive means the python instance is live somewhere
    ALIVE = "ALIVE"
    # sleeping means the db has all the instance info,
    # but the instance itself is not live.
    SLEEPING = "SLEEPING"
    # batch completed means its alive, and in its last run
    # has completed a batch.
    BATCH_COMPLETED = "BATCH_COMPLETED"

    # This is when a run is active.
    RUNNING = "RUNNING"

    #These are states that we can resume from 
    PAUSED = "PAUSED"
    FAILED = "FAILED"


class ToolRunSubStatus(str, Enum):
    # TODO: Implement
    ...


# Aliasing tool run status as agent run status
AgentRunStatus = ToolRunStatus
