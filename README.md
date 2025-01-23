# occam-core

This repository contains the open-core functionality for Occam AI's tools and
agents. It includes Pydantic-based models, helper functions, and utilities that
define how agents receive, process, and produce data.

## Overview

Occam agents rely on two main categories of Pydantic models:

1. **IO Models**
   These govern an agent's inputs and outputs.
   • All IO models inherit from the base [`IOModel`](./src/python/occam_core/util/base_models.py),
     which in turn inherits from `OccamDataType`.
   • The `OccamDataType` base class provides a system for recursively converting
     between different input/output structures. This is especially useful when
     chaining multiple agents, ensuring that the outputs of one agent can be
     adapted as the inputs of another.

2. **Agent Params Models**
   These specify configuration details for creating and running agents.
   • All parameters models inherit from
     [`AgentInstanceParamsModel`](./src/python/occam_core/util/base_models.py)
     (or one of its specialized subclasses).
   • Each agent's parameters model describes how that agent should be
     instantiated—a language agent might specify model name, system prompts, and
     logging preferences, while a human agent might specify contact and
     permission details.

In addition, the class
[`AgentIdentityCoreModel`](./src/python/occam_core/agents/model.py#L17)
encapsulates identifying information about each agent:
- The agent's name
- Contact information
- The name of the parameters model required to instantiate it

### Key Components

1. **Agents**
   Defined in `src/python/occam_core/agents/`.
   Agents use:
   - A `params.py` model (e.g., `LLMAgentParamsModel`) that details how the
     agent will operate.
   - An IO model (e.g., `AgentIOModel`) that describes the structure of
     incoming commands and outgoing responses.

2. **Utilities**
   Stored in `src/python/occam_core/util/`.
   - **`base_models.py`** houses the foundation for all IO and parameter models
     (`IOModel`, `AgentInstanceParamsModel`).
   - **`data_types/`** contains advanced Pydantic data types and the
     `OccamDataType` base, which offers recursive transformation of models.
   - **`common.py`** implements common methods like obtaining a project-level
     logger.

3. **Model Catalogue**
   In `model_catalogue.py`, the repository registers known parameter and IO
   models to facilitate setting up agent parameters.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/OccamAI/occam-core.git
   ```
2. Change into the directory:
   ```
   cd occam-core
   ```
3. Install dependencies using Poetry:
   ```
   poetry install
   ```
