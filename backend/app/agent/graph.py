from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.engines.analytical import run_analytical_engine
from app.engines.semantic import query_pdf, ingest_pdf
from app.core.config import settings
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=settings.google_api_key,
    temperature=0
)


# NODE 1: ROUTER
# Reads the question and decides which engine

def router_node(state: AgentState) -> AgentState:
    """
    Looks at the question and file (if any) and decides:
    - "csv" if the question is about structured/tabular data
    - "pdf" if the question is about documents/text content
    
    If a file is attached, it uses the file type to decide.
    If no file, it asks Gemini to classify the question.
    """
    question = state["question"]
    filename = state.get("filename", "")

    # If a file was uploaded, use file extension to route
    if filename:
        if filename.lower().endswith(".csv"):
            return {**state, "engine": "csv"}
        elif filename.lower().endswith(".pdf"):
            # Ingest the PDF first if file bytes are present
            if state.get("file_bytes"):
                ingest_pdf(state["file_bytes"], filename)
            return {**state, "engine": "pdf"}

    # No file uploaded - ask Gemini to classify the question
    prompt = f"""
    Classify this question into one of two categories:
    - "csv": Questions about data, numbers, statistics, spreadsheets, calculations, rows, columns
    - "pdf": Questions about documents, policies, text content, information lookup

    Question: {question}
    
    Reply with ONLY the single word: csv or pdf
    """

    response = llm.invoke(prompt)
    engine = response.content.strip().lower()

    if engine not in ["csv", "pdf"]:
        engine = "pdf"  # Default to PDF for ambiguous questions

    return {**state, "engine": engine}



# NODE 2: CSV ENGINE
# Handles structured data questions

def csv_node(state: AgentState) -> AgentState:
    """
    Runs the analytical engine on the uploaded CSV.
    """
    if not state.get("file_bytes"):
        return {
            **state,
            "status": "error",
            "error": "No CSV file provided. Please upload a CSV file with your question."
        }

    result = run_analytical_engine(
        state["file_bytes"],
        state["question"]
    )

    if result["status"] == "success":
        return {
            **state,
            "answer": result["answer"],
            "status": "success"
        }
    else:
        return {
            **state,
            "status": "error",
            "error": result.get("error", "CSV analysis failed")
        }



# NODE 3: PDF ENGINE
# Handles document/text questions

def pdf_node(state: AgentState) -> AgentState:
    """
    Runs the semantic RAG engine against stored documents.
    """
    result = query_pdf(
        state["question"],
        state.get("session_id", "default")
    )

    if result["status"] == "success":
        return {
            **state,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "status": "success"
        }
    else:
        return {
            **state,
            "status": "error",
            "error": result.get("error", "PDF query failed")
        }



# ROUTING FUNCTION
# Tells LangGraph which node to go to next

def route_to_engine(state: AgentState) -> str:
    """
    This function is called after router_node.
    It returns the name of the next node to execute.
    """
    return state.get("engine", "pdf")



# BUILD THE GRAPH

def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("csv", csv_node)
    graph.add_node("pdf", pdf_node)

    # Set entry point
    graph.set_entry_point("router")

    # Add conditional routing after router node
    graph.add_conditional_edges(
        "router",
        route_to_engine,
        {
            "csv": "csv",
            "pdf": "pdf"
        }
    )

    # Both engines go to END after completing
    graph.add_edge("csv", END)
    graph.add_edge("pdf", END)

    return graph.compile()


# Compile once at startup
smartops_graph = build_graph()