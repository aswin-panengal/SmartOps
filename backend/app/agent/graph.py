from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.engines.analytical import run_analytical_engine
from app.engines.semantic import query_pdf, ingest_pdf
from app.core.config import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agent.schemas import AssistantResponse

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=settings.google_api_key,
    temperature=0
)

# NODE 1: ROUTER
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
    session_id = state.get("session_id", "default")

    # If a file was uploaded, use file extension to route
    if filename:
        if filename.lower().endswith(".csv"):
            return {**state, "engine": "csv"}
        elif filename.lower().endswith(".pdf"):
            # Ingest the PDF first if file bytes are present, passing the session_id for isolation
            if state.get("file_bytes"):
                ingest_pdf(state["file_bytes"], filename, session_id)
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
def csv_node(state: AgentState) -> AgentState:
    """
    Runs the analytical engine using stateful session caching.
    """
    # Removed the 'if not file_bytes' error block because the analytical engine 
    # will now intelligently fetch the cached DataFrame using the session_id.
    result = run_analytical_engine(
        state.get("file_bytes"),
        state["question"],
        state.get("session_id", "default")
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
            "chunks_used": result.get("chunks_used", 0),
            "status": "success"
        }
    else:
        return {
            **state,
            "status": "error",
            "error": result.get("error", "PDF query failed")
        }

# NODE 4: THE COMPOSER
# Polishes the raw engine output into a friendly, structured assistant response
def composer_node(state: AgentState) -> AgentState:
    """
    Takes the raw answer from the semantic or analytical engine and uses the LLM
    to format it into a friendly, structured response with follow-up suggestions.
    """
    # If there was an error in the engine, just pass it through directly
    if state.get("status") == "error":
        return state

    raw_answer = state.get("answer", "")
    question = state.get("question", "")
    engine = state.get("engine", "unknown")

    # Use the structured output feature of Gemini
    structured_llm = llm.with_structured_output(AssistantResponse)

    prompt = f"""
    You are SmartOps, a helpful AI data assistant. 
    A backend engine just processed the user's question. 
    
    User Question: {question}
    Engine Used: {engine.upper()}
    Raw Engine Output:
    {raw_answer}
    
    Take the raw output above and format it into a friendly response.
    1. 'direct_answer': Provide the exact answer cleanly. If the raw output is a markdown table or code block, preserve it exactly.
    2. 'explanation': Briefly explain how you got the answer (e.g., "I summed the sales column" or "Based on the remote work policy document").
    3. 'follow_up_questions': Suggest 3 logical next questions the user could ask to explore this data/document further.
    """

    try:
        response_data = structured_llm.invoke(prompt)
        
        # Format the final string exactly how the frontend expects it
        final_markdown = f"{response_data.direct_answer}\n\n"
        final_markdown += f"*_{response_data.explanation}_*\n\n"
        final_markdown += "**Suggested Next Steps:**\n"
        for q in response_data.follow_up_questions:
            final_markdown += f"- {q}\n"

        return {
            **state,
            "answer": final_markdown
        }
    except Exception as e:
        # Fallback if structured output fails
        return state
    
    
# ROUTING FUNCTION
def route_to_engine(state: AgentState) -> str:
    """
    This function is called after router_node to determine the next graph route.
    """
    return state.get("engine", "pdf")


# BUILD THE GRAPH
def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("csv", csv_node)
    graph.add_node("pdf", pdf_node)
    graph.add_node("composer", composer_node) 

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

    # Both engines now go to the composer to be polished
    graph.add_edge("csv", "composer") 
    graph.add_edge("pdf", "composer") 
    
    # The composer goes to the END
    graph.add_edge("composer", END)   

    return graph.compile()


# Compile once at startup
smartops_graph = build_graph()



