"""Errors raised when data exchanged between agents is missing or invalid."""


class AgentError(Exception):
    """An agent could not produce or accept valid data."""

    def __init__(self, agent, message):
        self.agent = agent
        super().__init__(f"{agent}: {message}")


class StateValidationError(AgentError):
    """A required field in the shared state is missing or malformed."""


class StructuredOutputError(AgentError):
    """The LLM did not return output matching the expected schema."""
