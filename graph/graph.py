from langgraph.graph import StateGraph, END
from graph.state import ResearchState
from graph.nodes import (
    planner_node,
    researcher_node,
    extractor_node,
    writer_node,
    memory_node,
)

def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner",    planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("extractor",  extractor_node)
    graph.add_node("writer",     writer_node)
    graph.add_node("memory",     memory_node)      

    graph.set_entry_point("planner")

    graph.add_edge("planner",    "researcher")
    graph.add_edge("researcher", "extractor")
    graph.add_edge("extractor",  "writer")
    graph.add_edge("writer",     "memory")         
    graph.add_edge("memory",     END)

    return graph.compile()