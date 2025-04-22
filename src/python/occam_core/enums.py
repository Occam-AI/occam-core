from enum import Enum


class ToolStatus(str, Enum):

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



class ToolDataType(str, Enum):
    """
    This list defines the different categories of data
    that can agent can send.
    """

    STREAMING_UPDATES = "STREAMING_UPDATES"
    """Internal updates sent back from the tool sometimes this won't exist."""

    RUN_STEP_COMPLETED = "RUN_STEP_COMPLETED"
    """Data sent back when the tool announces that a batch step has completed."""

    OUTPUT = "OUTPUT"
    """Data sent back when the tool announces that it's completed a full streaming/batch run"""

    PAUSED = "PAUSED"
    """Data sent back when the tool announces that it's paused"""


class BatchStepDataType(str, Enum):
    """
    This list defines the different types of data
    that a tool can send when in batch mode.
    """

    RUN_STEP_COMPLETED = "RUN_STEP_COMPLETED"
    """Data sent back when the tool announces that a batch step has completed."""

    PAUSED = "PAUSED"
    """Data sent back when the tool announces that it's paused"""


class StreamingStepDataType(str, Enum):
    """
    This list defines the different types of data
    that a tool can send.
    """

    STREAMING_UPDATES = "STREAMING_UPDATES"
    """Internal updates sent back from the tool sometimes this won't exist."""

    RUN_STEP_COMPLETED = "RUN_STEP_COMPLETED"
    """Data sent back when the tool announces that a batch step has completed."""

    PAUSED = "PAUSED"
    """Data sent back when the tool announces that it's paused"""
