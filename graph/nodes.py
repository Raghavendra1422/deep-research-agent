import os
import json
import time
from dotenv import load_dotenv
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from graph.state import ResearchState
from tools.search import search_web
from tools.memory import store_report

load_dotenv()

# ── Gemini LLM or grokllm ────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",   
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)

#llm = ChatGoogleGenerativeAI(
 #   model="gemini-1.5-flash-latest",
  #  google_api_key=os.getenv("GEMINI_API_KEY"),
   # temperature=0.3,
#)

# ── Planner Node ──────────────────────────────────────────
def planner_node(state: ResearchState) -> dict:
    print("\n[PLANNER] Breaking query into sub-topics...")

    prompt = f"""
You are a research planner. The user wants a deep research report on:
"{state['user_query']}"

Break this into exactly 5 specific sub-topics that together cover the full research.
Return ONLY a JSON array of 5 strings. No explanation. No markdown. Example:
["topic 1", "topic 2", "topic 3", "topic 4", "topic 5"]
"""

    response = llm.invoke(prompt)
    time.sleep(3)
    raw = response.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    sub_topics = json.loads(raw)

    print(f"[PLANNER] Sub-topics identified:")
    for i, topic in enumerate(sub_topics, 1):
        print(f"  {i}. {topic}")

    return {
        "current_step": "planning_done",
        "sub_topics":   sub_topics
    }

# ── Researcher Node ───────────────────────────────────────
def researcher_node(state: ResearchState) -> dict:
    print("\n[RESEARCHER] Searching for each sub-topic...")

    all_results = []

    for topic in state["sub_topics"]:
        print(f"  🔍 Searching: {topic}")
        results = search_web(topic + " India 2025")

        all_results.append({
            "topic":   topic,
            "results": results
        })

    print(f"\n[RESEARCHER] Total topics searched: {len(all_results)}")

    return {
        "current_step":       "researching_done",
        "raw_search_results": all_results
    }

# ── Extractor Node ────────────────────────────────────────
def extractor_node(state: ResearchState) -> dict:
    print("\n[EXTRACTOR] Extracting facts from search results...")

    extracted_facts = []

    for item in state["raw_search_results"]:
        topic   = item["topic"]
        results = item["results"]

        context = ""
        for r in results:
            context += f"Title: {r['title']}\n"
            context += f"Snippet: {r['snippet']}\n"
            context += f"Source: {r['link']}\n\n"

        prompt = f"""
You are a research analyst. Based on the search results below, extract key facts about:
"{topic}"

Search Results:
{context}

Return a JSON object with this exact structure:
{{
    "topic": "{topic}",
    "key_facts": ["fact 1", "fact 2", "fact 3", "fact 4", "fact 5"],
    "statistics": ["any numbers, percentages, market sizes found"],
    "sources": ["source urls"]
}}

Return ONLY the JSON. No explanation. No markdown.
"""

        response = llm.invoke(prompt)
        raw = response.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            facts = json.loads(raw)
        except:
            facts = {
                "topic":      topic,
                "key_facts":  ["Could not extract facts for this topic"],
                "statistics": [],
                "sources":    []
            }

        print(f"  ✅ Extracted facts for: {topic[:50]}...")
        extracted_facts.append(facts)

    print(f"\n[EXTRACTOR] Facts extracted for {len(extracted_facts)} topics")

    return {
        "current_step":    "extracting_done",
        "extracted_facts": extracted_facts
    }

# ── Writer Node ───────────────────────────────────────────
def writer_node(state: ResearchState) -> dict:
    print("\n[WRITER] Writing final report...")

    # Build context from all extracted facts
    facts_context = ""
    for item in state["extracted_facts"]:
        facts_context += f"\n## {item['topic']}\n"
        facts_context += "Key Facts:\n"
        for fact in item.get("key_facts", []):
            facts_context += f"- {fact}\n"
        facts_context += "\nStatistics:\n"
        for stat in item.get("statistics", []):
            facts_context += f"- {stat}\n"
        facts_context += "\nSources:\n"
        for src in item.get("sources", []):
            facts_context += f"- {src}\n"

    prompt = f"""
You are a senior research analyst. Write a comprehensive, professional deep research report on:
"{state['user_query']}"

Based on these extracted facts:
{facts_context}

Write the report in this exact markdown structure:

# Deep Research Report: {state['user_query']}

## Executive Summary
(3-4 sentences overview of the entire market)

## 1. Market Size & Growth Projections
(detailed section with facts and statistics)

## 2. Key Players & Competitive Landscape
(detailed section with facts and statistics)

## 3. Demand Drivers & Applications
(detailed section with facts and statistics)

## 4. Government Policies & Initiatives
(detailed section with facts and statistics)

## 5. Challenges & Strategic Outlook
(detailed section with facts and statistics)

## Conclusion
(3-4 sentences final summary and future outlook)

## Sources
(list all sources)

Make it professional, data-driven, and detailed. Use the statistics wherever relevant.
"""

    response = llm.invoke(prompt)
    report = response.content.strip()

    # Save report to file
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n[WRITER] ✅ Report written successfully!")
    print("[WRITER] 📄 Saved to: research_report.md")

    return {
        "current_step": "done",
        "final_report": report
    }
# ── Memory Node ───────────────────────────────────────────
def memory_node(state: ResearchState) -> dict:
    print("\n[MEMORY] Saving report to ChromaDB with HuggingFace embeddings...")

    store_report(
        query=state["user_query"],
        report=state["final_report"]
    )

    return {"current_step": "stored"}