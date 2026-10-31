"""Runtime settings for parallel research, read from environment variables.

RESEARCH_MAX_TASKS        how many research tasks the supervisor plans (1-8, default 4)
RESEARCH_MAX_CONCURRENCY  how many workers may run at the same time (1-8, default 4)
"""
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESEARCH_TASKS = 4
DEFAULT_RESEARCH_CONCURRENCY = 4
LIMIT_MIN, LIMIT_MAX = 1, 8


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    clamped = max(LIMIT_MIN, min(LIMIT_MAX, value))
    if clamped != value:
        logger.warning("%s=%d is outside %d-%d; using %d", name, value, LIMIT_MIN, LIMIT_MAX, clamped)
    return clamped


def max_research_tasks():
    return _env_int("RESEARCH_MAX_TASKS", DEFAULT_MAX_RESEARCH_TASKS)


def research_concurrency():
    return _env_int("RESEARCH_MAX_CONCURRENCY", DEFAULT_RESEARCH_CONCURRENCY)
