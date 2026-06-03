from dotenv import load_dotenv
import os
load_dotenv()

from graph.graph import build_graph
from tools.memory import search_report, report_exists

def run_research(query: str):
    graph = build_graph()

    initial_state = {
        "user_query":         query,
        "sub_topics":         [],
        "raw_search_results": [],
        "scraped_content":    [],
        "extracted_facts":    [],
        "final_report":       None,
        "current_step":       "start",
        "error":              None,
    }

    print(f"\n🚀 Starting research on: {query}\n")
    result = graph.invoke(initial_state)

    print(f"\n{'='*60}")
    print("📊 REPORT PREVIEW (first 500 chars):")
    print('='*60)
    print(result["final_report"][:500])
    print('='*60)
    print("\n✅ Full report saved to research_report.md")

    return result


def ask_question(question: str):
    """Ask a question against stored research"""
    print(f"\n🔍 Searching stored research for: {question}")

    chunks = search_report(question)

    if not chunks:
        print("❌ No relevant information found in stored research.")
        return

    print("\n📚 Relevant findings from stored research:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Result {i} ---")
        print(chunk[:300])
        print()


if __name__ == "__main__":
    # Run full research pipeline
    run_research("AI semiconductor market in India")

    # Ask questions against stored report
    print("\n" + "="*60)
    print("💬 QUERYING STORED RESEARCH")
    print("="*60)
    ask_question("What are the government policies for semiconductors?")
    ask_question("What is the market size and growth rate?")