"""Small JSON-file store for past research runs (shown in the web UI)."""
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

HISTORY_PATH = Path(__file__).resolve().parent.parent / "output" / "history.json"
MAX_ENTRIES = 50


def load_history():
    """Newest first. Never raises; a corrupt file reads as empty."""
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except FileNotFoundError:
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("History file unreadable (%s); ignoring it", exc)
        return []


def _evaluation_dict(evaluation):
    """Compact, JSON-safe view of the final evaluation for the history list."""
    if evaluation is None:
        return None
    return {
        "overall": evaluation.overall_score,
        "passed": evaluation.passed,
        "summary": evaluation.summary,
        "dimensions": [
            {"name": d.name, "score": d.score, "weight": d.weight, "notes": d.notes}
            for d in evaluation.dimensions
        ],
        "flags": [
            {"severity": f.severity, "code": f.code, "message": f.message, "evidence": f.evidence}
            for f in evaluation.flags
        ],
    }


def make_entry(state, pdf_path, elapsed):
    """Build a history record from a finished workflow state."""
    return {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "topic": state.get("question", ""),
        "approved": bool(state.get("approved")),
        "status": state.get("final_status", "unknown"),
        "revisions": state.get("revision_count", 0),
        "reviews": len(state.get("review_history", [])),
        "elapsed": round(elapsed, 1),
        "evaluation": _evaluation_dict(state.get("latest_evaluation")),
        "draft": state.get("draft", ""),
        "feedback": state.get("feedback", ""),
        "pdf_path": str(pdf_path) if pdf_path else "",
    }


def add_entry(entry):
    """Prepend an entry and persist. Failures are logged, not raised."""
    history = [entry] + load_history()
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_PATH.write_text(
            json.dumps(history[:MAX_ENTRIES], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save history: %s", exc)
    return history[:MAX_ENTRIES]
