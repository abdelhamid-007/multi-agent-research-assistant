import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found!")

# Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=api_key,
)


def test_llm():
    response = llm.invoke(
        "How can AI agents automate customer support for SaaS companies? "
    )

    print("\n===== GROQ RESPONSE =====")
    print(response.content)


if __name__ == "__main__":
    test_llm()