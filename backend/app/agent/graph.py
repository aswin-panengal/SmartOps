from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.engines.analytical import run_analytical_engine
from app.engines.semantic import query_pdf, ingest_pdf
from app.core.config import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agent.schemas import RouterDecision

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=settings.google_api_key,
    temperature=0
)

GREETINGS = {
    "hey", "hello", "hi", "hii", "test", "ping",
    "good morning", "good afternoon", "good evening"
}

AFFIRMATIONS = {
    "yes", "y", "sure", "do it", "please", "yes please",
    "ok", "okay", "yeah", "show me", "go ahead"
}

VAGUE_REQUESTS = {
    "analyze", "analyse", "analyze this", "analyse this",
    "summarize", "summarise", "summarize this", "summarise this",
    "what do you think", "help", "help me", "start"
}

# NODE 1: SMART ROUTER
def router_node(state: AgentState) -> AgentState:
    question = state["question"]
    filename = state.get("filename", "")
    session_id = state.get("session_id", "default")
    file_bytes = state.get("file_bytes")
    lower_q = question.strip().lower()

    from app.engines.analytical import has_active_dataframe
    
    # 1. Pull Chat History safely
    try:
        from app.memory.session import build_context_for_prompt
        chat_history = build_context_for_prompt(session_id)
    except (ImportError, ModuleNotFoundError, NameError):
        chat_history = ""

    # 2. Store uploaded files before any greeting/clarification response. The
    # frontend sends the file only once, so this must never be skipped.
    if file_bytes and filename:
        if filename.lower().endswith(".pdf"):
            ingest_result = ingest_pdf(file_bytes, filename, session_id)
            if ingest_result.get("status") != "success":
                return {
                    **state,
                    "engine": "clarify",
                    "answer": ingest_result.get(
                        "error",
                        "I could not read that PDF. Please try a text-based PDF or upload a clearer file."
                    )
                }
            if lower_q in GREETINGS:
                return {
                    **state,
                    "engine": "clarify",
                    "answer": "Hello, I have indexed your PDF. Ask me for a summary, key points, risks, or any detail from the document."
                }
            if lower_q in VAGUE_REQUESTS:
                return {**state, "engine": "pdf", "question": "Summarize this document and highlight the key points."}
            return {**state, "engine": "pdf"}
        elif filename.lower().endswith(".csv"):
            if lower_q in GREETINGS:
                return {
                    **state,
                    "engine": "csv",
                    "question": (
                        "Give a quick useful profile of this CSV: row count, column count, column names, "
                        "missing values, obvious numeric totals or averages, and any notable anomalies."
                    )
                }
            if lower_q in VAGUE_REQUESTS:
                return {
                    **state,
                    "engine": "csv",
                    "question": (
                        "Give a quick useful profile of this CSV: row count, column count, column names, "
                        "missing values, obvious numeric totals or averages, and any notable anomalies."
                    )
                }
            return {**state, "engine": "csv"}

    csv_loaded = has_active_dataframe(session_id)

    # 3. Fast deterministic handling for common low-value turns.
    if lower_q in GREETINGS:
        file_hint = (
            "Your CSV is ready, so you can ask for totals, trends, missing values, or anomalies."
            if csv_loaded
            else "Upload a CSV for data analysis or a PDF for document questions, then ask naturally."
        )
        return {
            **state,
            "engine": "clarify",
            "answer": f"Hello, I am ready to help. {file_hint}"
        }

    # 4. Automated turn-taking for short confirmations.
    # If the user gives an affirmative confirmation, look back at what the assistant suggested
    if csv_loaded and lower_q in AFFIRMATIONS:
        
        # Ask Gemini to quickly extract the exact analysis question from the last message's context
        context_prompt = f"""
        The user just said '{question}' to confirm a suggestion.
        Look at the recent conversation history below and determine what specific question or analysis the assistant offered to do.
        Return ONLY that specific analysis question as a plain sentence. Do not include any filler text.
        
        History:
        {chat_history}
        """
        try:
            # We use the raw LLM invoke to swap out the question dynamically
            resolved_question = llm.invoke(context_prompt).content.strip()
            return {
                **state,
                "question": resolved_question, # Overrides "yes" with the actual question!
                "engine": "csv"
            }
        except Exception:
            # Fallback if context extraction drops out
            return {**state, "engine": "csv"}

    if lower_q in VAGUE_REQUESTS:
        if csv_loaded:
            return {
                **state,
                "engine": "csv",
                "question": (
                    "Give a quick useful profile of the active CSV: row count, column count, column names, "
                    "missing values, obvious numeric totals or averages, and any notable anomalies."
                )
            }
        return {
            **state,
            "engine": "clarify",
            "answer": "Please upload a CSV or PDF first, then ask what you want to know from it."
        }

    # 5. Standard Router Execution for new inputs.
    structured_router = llm.with_structured_output(RouterDecision)
    
    prompt = f"""
    Evaluate the user's request to decide the routing engine.
    
    Recent Conversation History:
    {chat_history if chat_history else "None"}
    
    Current User Request: "{question}"
    CSV File Loaded in Memory: {csv_loaded}
    
    Rules:
    - If vague (e.g., "analyze this", "summarize"): set is_clear=False, engine="clarify", and write a helpful clarification_message offering specific options.
    - CRITICAL: If 'CSV File Loaded in Memory' is True, your clarification message MUST NOT say the file is missing. You must acknowledge that the data is ready and offer options!
    - If specific (e.g., "what is the total revenue"): set is_clear=True, and set engine to "csv" (data) or "pdf" (text).
    - If no history and it's a greeting, set engine="clarify" and say hello.
    """
    
    try:
        decision = structured_router.invoke(prompt)
        engine = decision.engine if decision.engine in {"csv", "pdf", "clarify"} else "clarify"

        if engine == "csv" and not csv_loaded:
            return {
                **state,
                "engine": "clarify",
                "answer": "I can do that once a CSV is uploaded. Please attach the CSV and ask again."
            }

        return {
            **state,
            "engine": engine,
            "answer": decision.clarification_message if not decision.is_clear else None
        }
    except Exception as e:
        # STRICT FALLBACK: Prevent 500 errors if the LLM provider rate limits or drops connection
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            safe_reply = "I am receiving too many requests right now and hit an API rate limit. Please wait about 60 seconds and try asking again."
        else:
            safe_reply = "My routing engine experienced a temporary connection issue. Please try again."
            
        return {
            **state,
            "engine": "clarify", # Route safely back to the user
            "answer": safe_reply
        }
# NODE 2: CSV ENGINE
def csv_node(state: AgentState) -> AgentState:
    """
    Runs the analytical engine using stateful session caching.
    """
    result = run_analytical_engine(
        state.get("file_bytes"),
        state["question"],
        state.get("session_id", "default")
    )

    if result["status"] == "success":
        try:
            from app.memory.session import add_message
            add_message(state.get("session_id", "default"), "user", state["question"])
            add_message(state.get("session_id", "default"), "assistant", result["answer"])
        except Exception:
            pass

        return {
            **state,
            "answer": result["answer"],
            "rows_in_file": result.get("rows_in_file"),
            "columns": result.get("columns", []),
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
            "contexts": result.get("contexts", []),
            "status": "success"
        }
    else:
        return {
            **state,
            "status": "error",
            "error": result.get("error", "PDF query failed")
        }

# NODE 4: THE COMPOSER
def composer_node(state: AgentState) -> AgentState:
    """
    Takes the raw answer from the semantic or analytical engine and formats it
    without another LLM call. This keeps answers fast and avoids rewriting tables.
    """
    if state.get("status") == "error":
        return state

    raw_answer = state.get("answer", "")
    engine = state.get("engine", "unknown")

    if engine == "csv":
        row_count = state.get("rows_in_file")
        explanation = (
            f"I checked this against the active CSV dataset ({row_count} rows)."
            if row_count
            else "I checked this against the active CSV dataset."
        )
        follow_up = "Want me to break it down by category, date, or top/bottom values?"
    elif engine == "pdf":
        sources = state.get("sources") or []
        source_text = f" Source: {', '.join(sources)}." if sources else ""
        explanation = f"I used the most relevant document chunks from your uploaded PDF.{source_text}"
        follow_up = "Want me to summarize the key points or check another section?"
    else:
        return state

    return {
        **state,
        "answer": f"{raw_answer}\n\n_{explanation}_\n\n{follow_up}"
    }
    
    
# ROUTING FUNCTION
def route_to_engine(state: AgentState) -> str:
    """
    This function is called after router_node to determine the next graph route.
    """
    engine = state.get("engine") or "clarify"
    return engine if engine in {"csv", "pdf", "clarify"} else "clarify"

# NODE: CLARIFY
def clarify_node(state: AgentState) -> AgentState:
    """
    Bypasses the processing engines and asks the user for more details.
    """
    return {
        **state,
        "answer": state.get("answer") or "I can help with CSV analysis or PDF questions. Please upload a file or ask a more specific question.",
        "status": "success" 
    }


# BUILD THE GRAPH
def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("csv", csv_node)
    graph.add_node("pdf", pdf_node)
    graph.add_node("composer", composer_node)

    graph.set_entry_point("router")

    # Add conditional routing
    graph.add_conditional_edges(
        "router",
        route_to_engine,
        {
            "csv": "csv",
            "pdf": "pdf",
            "clarify": "clarify" 
        }
    )

    # Wire up the new paths
    graph.add_edge("csv", "composer")
    graph.add_edge("pdf", "composer")
    
    graph.add_edge("composer", END)
    graph.add_edge("clarify", END) # Clarify goes straight to END, skipping composer

    return graph.compile()


# Compile once at startup
smartops_graph = build_graph()
