
import logging

from app.llm import llm
from app.schemas import AnalysisResult, require_research
from app.structured import invoke_structured

logger = logging.getLogger(__name__)

_FORMAT = """{
  "key_considerations": ["..."],
  "opportunities": ["business opportunity"],
  "technical_feasibility": "short assessment",
  "risks": ["..."],
  "implementation_complexity": "low | medium | high",
  "recommendations": ["..."],
  "assumptions": ["anything you state that the research does not support"]
}"""


def analysis_agent(state):

    print("\n📊 Analysis Agent is working...")

    research = require_research(state, "Analysis")

    prompt = f"""
    You are a professional business intelligence analyst.

    User question:
    {research.question or state.get('question', '')}

    Use the structured research below as your primary evidence. Source
    numbers like [1] refer to the numbered sources.

    {research.to_text(max_chars=5000)}

    Analyze the findings and answer the original question.
    Do not invent statistics. Anything not supported by the research must
    be listed under "assumptions".

    Reply with ONLY a JSON object in exactly this format:
    {_FORMAT}
    """

    analysis = invoke_structured(llm, prompt, AnalysisResult, "Analysis")

    logger.info(
        "Analysis: %d opportunity(ies), %d risk(s), complexity=%s",
        len(analysis.opportunities), len(analysis.risks), analysis.implementation_complexity,
    )

    return {
        "analysis_data": analysis,
        "analysis": analysis.to_text(),
        "execution_history": state.get("execution_history", []) + ["analysis"],
    }
