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

CSV_ACTION_REQUESTS = {
    "summary", "summarize", "summarise", "overview", "analysis",
    "analyze", "analyse", "profile", "quick summary", "quick overview",
    "missing values", "anomaly", "anomalies", "anomaly check",
    "trends", "trend", "top values", "bottom values"
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

    if csv_loaded and lower_q in CSV_ACTION_REQUESTS:
        return {
            **state,
            "engine": "csv",
            "question": (
                f"User selected this analysis task for the active CSV: {question}. "
                "Answer it directly using the loaded dataframe."
            )
        }

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
    1. Conversational continuity / turn-taking:
       - Always read Recent Conversation History before judging the current request.
       - If the assistant's last message offered specific options such as summary,
         overview, anomaly check, missing values, trends, top/bottom values, or
         document sections, and the Current User Request is a short selection of
         one of those options, treat it as a specific request.
       - In that case set is_clear=True. Do NOT ask for clarification again.
       - Route to csv if the selected option is about data/table analysis.
       - Route to pdf if the selected option is about document/PDF content.

    2. Bias towards action:
       - If CSV File Loaded in Memory is True and the user asks for a general
         "summary", "overview", "analysis", "analyze", "profile", "missing
         values", "trends", or "anomaly check", set is_clear=True and
         engine="csv". The pandas engine can handle broad CSV analysis.
       - Do not route broad CSV requests to clarify when a CSV is loaded.

    3. File awareness:
       - If CSV File Loaded in Memory is True, any clarification_message must
         never claim that no CSV, dataset, data, table, or file is provided.
       - If clarification is truly needed while CSV is loaded, acknowledge that
         the CSV is ready and ask which analysis to run.

    4. Missing file behavior:
       - If the user asks for CSV/data analysis and CSV File Loaded in Memory is
         False, set is_clear=False, engine="clarify", and ask them to upload a CSV.
       - If the user asks a document/PDF question without useful PDF context in
         history, choose pdf only if the request is clearly document-oriented.

    5. Specific requests:
       - If specific (e.g., "what is the total revenue", "which region has the
         highest sales", "what does the policy say about leave"), set
         is_clear=True and choose csv for tabular/data questions or pdf for
         document/text questions.

    6. Greetings:
       - If no file context and the user is only greeting, set is_clear=False,
         engine="clarify", and briefly invite them to upload a CSV or PDF.
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
            "retrieval_scores": result.get("retrieval_scores", []),
            "best_score": result.get("best_score"),
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
    question = state.get("question", "").lower()

    def optional_follow_up() -> str:
        """
        Keep conversation clean. Ask a follow-up only when the user's current
        question naturally points to a next step; otherwise stay quiet.
        """
        if engine == "csv":
            if any(word in question for word in ["summary", "overview", "profile", "analysis", "analyze", "analyse"]):
                return "Do you want me to check missing values or unusual records next?"
            if any(word in question for word in ["total", "sum", "revenue", "sales", "amount"]):
                return "Do you want the same metric grouped by category or date?"
            if any(word in question for word in ["top", "highest", "best", "maximum", "max"]):
                return "Do you want to see the bottom values too?"
            if any(word in question for word in ["bottom", "lowest", "worst", "minimum", "min"]):
                return "Do you want to see the top values too?"
            if any(word in question for word in ["missing", "null", "empty", "blank"]):
                return "Do you want me to show the rows affected by those missing values?"
            return ""

        if engine == "pdf":
            if any(word in question for word in ["summary", "summarize", "summarise", "overview", "key points"]):
                return "Do you want me to list the risks or action items from it?"
            if any(word in question for word in ["policy", "rule", "requirement", "condition"]):
                return "Do you want me to check the exceptions or related clauses?"
            if any(word in question for word in ["risk", "issue", "problem"]):
                return "Do you want me to turn those into action items?"
            return ""

        return ""

    if engine == "csv":
        row_count = state.get("rows_in_file")
        explanation = (
            f"I checked this against the active CSV dataset ({row_count} rows)."
            if row_count
            else "I checked this against the active CSV dataset."
        )
    elif engine == "pdf":
        sources = state.get("sources") or []
        source_text = f" Source: {', '.join(sources)}." if sources else ""
        explanation = f"I used the most relevant document chunks from your uploaded PDF.{source_text}"
    else:
        return state

    follow_up = optional_follow_up()
    final_answer = f"{raw_answer}\n\n_{explanation}_"
    if follow_up:
        final_answer += f"\n\n{follow_up}"

    return {
        **state,
        "answer": final_answer
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
    answer = state.get("answer") or "I can help with CSV analysis or PDF questions. Please upload a file or ask a more specific question."
    try:
        from app.memory.session import add_message
        add_message(state.get("session_id", "default"), "user", state.get("question", ""))
        add_message(state.get("session_id", "default"), "assistant", answer)
    except Exception:
        pass

    return {
        **state,
        "answer": answer,
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
