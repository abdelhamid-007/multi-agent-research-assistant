
import time
from concurrent.futures import ThreadPoolExecutor


def research_agent():
    print("Research Agent started...")
    time.sleep(3)
    print("Research Agent finished.")
    return "Research findings"


def analysis_agent():
    print("Analysis Agent started...")
    time.sleep(3)
    print("Analysis Agent finished.")
    return "Analysis findings"


if __name__ == "__main__":
    start = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        research_future = executor.submit(research_agent)
        analysis_future = executor.submit(analysis_agent)

        research = research_future.result()
        analysis = analysis_future.result()

    print("\nResults:")
    print(research)
    print(analysis)

    print(f"\nExecution time: {time.time() - start:.2f} seconds")
