from enum import Enum


class ToolRunStatus(str, Enum):
    """
    new
    ALIVE: 
    BATCH_COMPLETED:
    FAILED: we have batch number (with failure type)
    RUNNING: we have batch number (with )
    PAUSED: 

    run sub-statuses
    Big enum that overlaps across run statuses. Mainly relevant for failures and pauses.

    Think of a number of classes/abstractions for failures and appropriate messages.
    Create an abstraction that returns messages.

    Can be children of AgentIOModel
    - has run_result field
    - request for modifications
    - type of model (relates to topic of models)

    Think of errors as AgentIOModel

    Even AgentInstanceMetadata can be conceptualised as one of those messages

    Move polling for status out of checkpointer.

    Side but IMPORTANT note: Agent parameterisation and how it will work from Autogen.

    User agent and multi-agent workspaces instantiation has some complexities in parameterisation:
    - Budget and other "meta" parameters.
    - 
    """
    ALIVE = "ALIVE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    BATCH_COMPLETED = "BATCH_COMPLETED"
    FAILED = "FAILED"


class ToolRunSubStatus(str, Enum):
    # TODO: Implement
    ...


# Aliasing tool run status as agent run status
AgentRunStatus = ToolRunStatus
