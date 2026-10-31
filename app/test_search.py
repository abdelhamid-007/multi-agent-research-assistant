
from dotenv import load_dotenv

load_dotenv()

from app.tools.web_search import search_web


def main():

    results = search_web(
        "Latest AI agent development trends in 2026"
    )

    print("\n========== SEARCH RESULTS ==========\n")

    print(results)


if __name__ == "__main__":
    main()
