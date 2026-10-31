
import logging

from app.reports import export_report_pdf
from app.runner import build_initial_state, graph_config
from app.workflows.graph import research_graph


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    initial_state = build_initial_state("Latest AI agent development trends in 2026")

    print("\n🚀 Starting Supervisor Multi-Agent System")

    result = research_graph.invoke(
        initial_state,
        config=graph_config()
    )

    print("\n========== FINAL REPORT ==========\n")
    print(result["draft"])

    print("\n========== EXECUTION HISTORY ==========\n")

    for agent in result["execution_history"]:
        print(" ->", agent)

    print("\n========== FINAL STATUS ==========")

    if result["approved"]:
        print("✅ Report approved!")
    else:
        print("⚠️ Report NOT approved - treat it as an unreviewed draft.")

    print("Status:", result["final_status"])
    print("Reviews:", len(result.get("review_history", [])))
    print("Revisions:", result["revision_count"])

    print("\nFinal reviewer feedback:")
    print(result["feedback"])

    evaluations = result.get("evaluation_history", [])
    if evaluations:
        print("\n========== EVALUATION ==========")
        print(result["latest_evaluation"].summary)
        print("Overall score per review:",
              ", ".join(f"revision {e.revision} = {e.overall_score:.0f}" for e in evaluations))

    # PDF export runs after the workflow; it never raises and never alters
    # the workflow result.
    pdf_path = export_report_pdf(result)

    print("\n========== PDF REPORT ==========")
    if pdf_path:
        print("📄 Saved:", pdf_path)
    else:
        print("⚠️ PDF could not be generated (see log for details).")


if __name__ == "__main__":
    main()

