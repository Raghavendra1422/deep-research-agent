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
    print("🤖 Deep Research AI Agent")
    print("="*40)
    print("Powered by LangGraph + Groq + Serper + HuggingFace + ChromaDB")
    print("="*40)

    # Get query from user
    query = input("\n🔍 Enter your research topic: ").strip()
    if not query:
        query = "AI semiconductor market in India"
        print(f"No input given. Using default: {query}")

    # Run full research pipeline
    run_research(query)

    # Interactive Q&A loop
    print("\n" + "="*60)
    print("💬 ASK QUESTIONS ABOUT YOUR RESEARCH")
    print("    Type 'exit' to quit")
    print("="*60)

    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("\n👋 Exiting. Your report is saved in research_report.md")
            break
        if not question:
            continue
        ask_question(question)