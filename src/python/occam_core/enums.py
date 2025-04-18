from enum import Enum


class ToolRunStatus(str, Enum):

    """
    These are run ready states.
    """
    # alive means the python instance is live somewhere
    # but hasn't run yet.
    ALIVE = "ALIVE"
    # sleeping means the db has all the instance info,
    # but the instance itself is not live.
    SLEEPING = "SLEEPING"
    # batch completed means its alive, and in its last run
    # has completed a batch.
    BATCH_COMPLETED = "BATCH_COMPLETED"

    # Request States
    PAUSE_REQUESTED = "PAUSE_REQUESTED"
    RESUME_REQUESTED = "RESUME_REQUESTED"
    TERMINATE_REQUESTED = "TERMINATE_REQUESTED"

    # Request in progress states
    PAUSE_IN_PROGRESS = "PAUSE_IN_PROGRESS"
    RESUME_IN_PROGRESS = "RESUME_IN_PROGRESS"
    TERMINATE_IN_PROGRESS = "TERMINATE_IN_PROGRESS"

    # Run related states
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    TERMINATED = "TERMINATED"


class ToolRunSubStatus(str, Enum):
    # TODO: Implement
    ...
