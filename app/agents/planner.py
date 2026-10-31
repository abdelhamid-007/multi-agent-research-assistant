from app.llm import llm


def planner_agent(state):

    print("\n🧠 Planner Agent is working...")

    prompt = f"""
    You are an expert research planner.

    Your task is to create a detailed research plan
    for the following question:

    {state['question']}

    Break the task into 5 specific research steps.

    Make each step clear and actionable.

    Return only the research plan.
    """

    response = llm.invoke(prompt)

    return {
        "plan": response.content
    }