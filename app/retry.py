"""Retry with exponential backoff for flaky LLM / search calls."""
import logging
import random
import re
import time

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 4
BASE_DELAY = 2.0
MAX_DELAY = 60.0
# A provider asking for a longer wait than this (e.g. a daily quota) will not
# recover within a normal retry, so fail fast instead of burning minutes.
MAX_SUGGESTED_WAIT = 120.0

_TRANSIENT_CODES = {408, 409, 413, 429, 500, 502, 503, 504}
_TRANSIENT_WORDS = (
    "rate limit", "rate_limit", "too many requests", "timed out", "timeout",
    "connection", "temporarily", "overloaded", "service unavailable",
)


def is_transient(exc):
    """True for errors worth retrying (rate limits, 5xx, network)."""
    code = getattr(exc, "status_code", None)
    if code is not None:
        return code in _TRANSIENT_CODES
    text = f"{type(exc).__name__} {exc}".lower()
    if re.search(r"error code: (408|409|413|429|5\d\d)", text):
        return True
    return any(word in text for word in _TRANSIENT_WORDS)


def _suggested_wait(exc):
    """Groq says e.g. 'Please try again in 7.5s' / '1m12s'; honour it."""
    m = re.search(r"try again in (?:(\d+)m)?\s*([\d.]+)s", str(exc))
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def _sleep(seconds):
    time.sleep(seconds)


def retry_call(fn, *args, attempts=MAX_ATTEMPTS, label="call", sleep=None, **kwargs):
    """Call fn(*args, **kwargs), retrying transient failures with backoff.

    Non-transient errors (bad key, invalid request, bugs) are raised at once;
    after the last attempt the original exception is raised.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts or not is_transient(exc):
                if attempt > 1:
                    logger.error("%s failed after %d attempt(s): %s", label, attempt, exc)
                raise
            suggested = _suggested_wait(exc)
            if suggested and suggested > MAX_SUGGESTED_WAIT:
                logger.error(
                    "%s: the provider asks for a %.0fs wait (e.g. a daily quota); not retrying: %s",
                    label, suggested, str(exc)[:200],
                )
                raise
            delay = min(BASE_DELAY * 2 ** (attempt - 1), MAX_DELAY)
            delay = max(delay, min(_suggested_wait(exc) or 0, MAX_DELAY)) + random.uniform(0, 1)
            logger.warning(
                "%s hit a transient error (attempt %d/%d): %s - retrying in %.1fs",
                label, attempt, attempts, str(exc)[:160], delay,
            )
            (sleep or _sleep)(delay)


def invoke_llm(llm, prompt, label="LLM call"):
    return retry_call(llm.invoke, prompt, label=label)
