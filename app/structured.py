"""Get schema-validated JSON out of any LLM (no provider features needed)."""
import json
import logging
import re

from pydantic import ValidationError

from app.errors import StructuredOutputError
from app.retry import invoke_llm

logger = logging.getLogger(__name__)


def extract_json(text):
    """Pull the first JSON object out of an LLM reply (fences/preamble ok)."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    return text[start:end + 1]


def _parse(text, model, validate=None):
    data = json.loads(extract_json(text))
    obj = model.model_validate(data)
    if validate:
        validate(obj)
    return obj


def _describe(exc):
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'root'}: {e['msg']}" for e in exc.errors()[:8]
        )
    return str(exc)


def invoke_structured(llm, prompt, model, agent, validate=None, repairs=1):
    """Invoke the LLM and return a validated `model` instance.

    On invalid output the model gets `repairs` chances to fix it, shown its
    own error. `validate(obj)` may raise ValueError for cross-field rules the
    schema cannot know (e.g. source ids that must exist in the research).
    Raises StructuredOutputError if it never becomes valid.
    """
    current = prompt
    last_error = "unknown"
    for attempt in range(repairs + 1):
        reply = invoke_llm(llm, current, f"{agent} LLM call").content
        try:
            return _parse(reply, model, validate)
        except (ValueError, ValidationError) as exc:  # JSONDecodeError is a ValueError
            last_error = _describe(exc)
            logger.warning("%s: invalid structured output (attempt %d/%d): %s",
                           agent, attempt + 1, repairs + 1, last_error)
            current = (
                f"{prompt}\n\nYour previous reply was rejected: {last_error}\n"
                "Reply again with ONLY the corrected JSON object."
            )
    raise StructuredOutputError(
        agent, f"the model did not return valid output after {repairs + 1} attempt(s): {last_error}"
    )
