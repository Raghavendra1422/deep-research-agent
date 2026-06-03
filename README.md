# 🔬 Deep Research AI Agent

An autonomous AI agent that conducts deep, multi-step research on any topic and generates a comprehensive professional report — powered by **LangGraph**, **Groq LLM**, **Serper API**, **HuggingFace Embeddings**, and **ChromaDB**.

---

## 🧠 Problem Statement

When you ask a plain LLM like ChatGPT to research a complex topic like *"AI Semiconductor Market in India"*, it fails in several ways:

- **Knowledge cutoff** — it doesn't know recent news, investments, or policy updates
- **One-shot thinking** — it cannot search → read → decide → search again like a real analyst
- **No tool use** — it cannot browse the web, scrape pages, or verify facts
- **No memory** — it cannot store findings and answer follow-up questions instantly

This project solves all of these problems using an **agentic AI pipeline**.

---

## ✅ Solution — Agentic Research Pipeline

Instead of one LLM call, this system uses a **LangGraph state machine** with specialized nodes that reason in a loop:

```
User Query
    ↓
[Planner Agent]     → Breaks query into 5 focused sub-topics
    ↓
[Researcher Agent]  → Searches web via Serper API for each sub-topic
    ↓
[Extractor Agent]   → Pulls structured facts, stats, and sources from results
    ↓
[Writer Agent]      → Writes a professional markdown research report
    ↓
[Memory Agent]      → Stores report in ChromaDB using HuggingFace embeddings
    ↓
Semantic Q&A        → User can query the stored report instantly
```

---

## 🏗️ Architecture

```
deep_research_agent/
│
├── main.py                  # Entry point + Q&A interface
├── graph/
│   ├── state.py             # Shared LangGraph state (TypedDict)
│   ├── nodes.py             # All 5 agent nodes
│   └── graph.py             # LangGraph graph builder
├── tools/
│   ├── search.py            # Serper API web search tool
│   ├── scraper.py           # BeautifulSoup scraper
│   └── memory.py            # ChromaDB + HuggingFace vector memory
├── research_report.md       # Generated report output
├── .env                     # API keys (not committed)
├── .gitignore
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Agent Orchestration | **LangGraph** | State machine, node graph, agent flow |
| LLM Brain | **Groq (LLaMA 3.3 70B)** | Planning, extraction, report writing |
| Web Search | **Serper API** | Real-time Google search results |
| Web Scraping | **BeautifulSoup** | Raw page content extraction |
| Embeddings | **HuggingFace (all-MiniLM-L6-v2)** | Local, free, semantic embeddings |
| Vector Store | **ChromaDB** | Persistent local vector database |
| Report Output | **Markdown** | Professional formatted research report |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Raghavendra1422/deep-research-agent.git
cd deep-research-agent
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

Get your free API keys:
- **Groq** → [console.groq.com](https://console.groq.com) (free, no credit card)
- **Serper** → [serper.dev](https://serper.dev) (100 free searches)

### 5. Run the agent
```bash
python main.py
```

---

## 📊 Example Output

**Query:** `AI semiconductor market in India`

```
🚀 Starting research on: AI semiconductor market in India

[PLANNER] Sub-topics identified:
  1. Introduction to AI Semiconductor Market in India
  2. Current Market Trends and Size
  3. Key Players and Competitive Landscape
  4. Government Initiatives and Regulatory Framework
  5. Future Outlook and Growth Prospects

[RESEARCHER] Searching each sub-topic via Serper API...
[EXTRACTOR] Extracting structured facts from search results...
[WRITER] Writing professional research report...
[MEMORY] Storing report in ChromaDB with HuggingFace embeddings...

✅ Report saved to research_report.md

💬 QUERYING STORED RESEARCH
🔍 What are the government policies for semiconductors?
→ The Indian government launched ISM with ₹76,000 crore outlay...
```

---

## 💡 Key Concepts Demonstrated

- **LangGraph State Machine** — nodes share state via `TypedDict`, each node reads and writes to a central state object
- **Agentic Reasoning** — multi-step pipeline where each agent's output feeds the next
- **RAG on Agent Output** — report is chunked, embedded, and stored so users can do semantic Q&A on it
- **Local Embeddings** — HuggingFace `all-MiniLM-L6-v2` runs fully offline, no API cost
- **Persistent Vector Store** — ChromaDB stores embeddings locally across sessions

---

## 🔑 Why Agents Over Plain LLM?

| Problem with plain LLM | How this agent solves it |
|---|---|
| Knowledge cutoff | Serper fetches real-time web results |
| One-shot answer | Multi-node pipeline reasons step by step |
| No tool use | Agent uses search + scrape + embed tools |
| No memory | ChromaDB stores and retrieves findings |
| Hallucinated stats | Facts extracted from real sources with citations |

---

## 📈 Future Improvements

- [ ] Add BeautifulSoup full page scraping for deeper content
- [ ] Add PDF ingestion for government policy documents (PyMuPDF)
- [ ] Add a Streamlit/FastAPI frontend for interactive Q&A
- [ ] Add LangGraph conditional edges for retry on failed searches
- [ ] Switch to Gemini API for production deployment

---

## 👤 Author

**Chelimela Raghavendra Goud**
- GitHub: [@Raghavendra1422](https://github.com/Raghavendra1422)
- LinkedIn: [raghavendra1422](https://linkedin.com/in/raghavendra1422)
