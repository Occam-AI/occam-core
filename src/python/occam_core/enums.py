from enum import Enum


class ToolRunStatus(str, Enum):
    #These are states where the instance exists
    # but no run is active.
    # alive means the python instance is live somewhere
    ALIVE = "ALIVE"
    # sleeping means the db has all the instance info,
    # but the instance itself is not live.
    SLEEPING = "SLEEPING"

    # When a pause has been requested, but not yet processed.
    PAUSE_REQUESTED = "PAUSE_REQUESTED"
    RESUME_REQUESTED = "RESUME_REQUESTED"
    TERMINATE_REQUESTED = "TERMINATE_REQUESTED"

    # in progress signals
    PAUSE_IN_PROGRESS = "PAUSE_IN_PROGRESS"
    RESUME_IN_PROGRESS = "RESUME_IN_PROGRESS"
    TERMINATE_IN_PROGRESS = "TERMINATE_IN_PROGRESS"

    #These are states that we can resume from 
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    TERMINATED = "TERMINATED"

    # batch completed means its alive, and in its last run
    # has completed a batch.
    BATCH_COMPLETED = "BATCH_COMPLETED"

class ToolRunSubStatus(str, Enum):
    # TODO: Implement
    ...
