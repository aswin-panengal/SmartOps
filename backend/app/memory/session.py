import time
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=settings.google_api_key,
    temperature=0,
)

# Key: session_id → {"messages": [...], "summary": str|None, "last_active": float}
sessions: dict[str, dict] = {}

MAX_MESSAGES_BEFORE_SUMMARY = 10
SESSION_TTL_SECONDS = 3600        # evict sessions idle for > 1 hour
MAX_SESSIONS = 500                # hard cap to prevent unbounded growth


def _evict_stale_sessions() -> None:
    """Remove sessions that have been idle past TTL or when the cap is hit."""
    now = time.time()
    stale = [sid for sid, s in sessions.items() if now - s["last_active"] > SESSION_TTL_SECONDS]
    for sid in stale:
        sessions.pop(sid, None)
        try:
            from app.engines.analytical import clear_active_dataframe
            clear_active_dataframe(sid)
        except Exception:
            pass

    # If still over cap after TTL eviction, remove oldest by last_active
    if len(sessions) >= MAX_SESSIONS:
        sorted_ids = sorted(sessions, key=lambda sid: sessions[sid]["last_active"])
        for sid in sorted_ids[: len(sessions) - MAX_SESSIONS + 1]:
            sessions.pop(sid, None)


def get_or_create_session(session_id: str) -> dict:
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            _evict_stale_sessions()
        sessions[session_id] = {
            "messages": [],
            "summary": None,
            "last_active": time.time(),
        }
    else:
        sessions[session_id]["last_active"] = time.time()
    return sessions[session_id]


def add_message(session_id: str, role: str, content: str) -> None:
    session = get_or_create_session(session_id)
    session["messages"].append({"role": role, "content": content})

    if len(session["messages"]) > MAX_MESSAGES_BEFORE_SUMMARY:
        _summarize_old_messages(session_id)


def _summarize_old_messages(session_id: str) -> None:
    """
    Summarize the older half of the conversation. Runs synchronously but is
    only triggered once every MAX_MESSAGES_BEFORE_SUMMARY turns, so the
    amortized cost per message is low. Failures are logged and swallowed —
    they must not block the caller.
    """
    session = sessions.get(session_id)
    if not session:
        return

    messages = session["messages"]
    midpoint = len(messages) // 2
    old_messages = messages[:midpoint]
    session["messages"] = messages[midpoint:]

    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in old_messages
    )

    try:
        summary_prompt = (
            "Summarize this conversation history concisely. "
            "Preserve all important facts, decisions, and context.\n\n"
            f"Conversation:\n{conversation_text}\n\nSummary:"
        )
        response = llm.invoke(summary_prompt)
        new_summary = response.content

        if session["summary"]:
            new_summary = f"{session['summary']}\n\n{new_summary}"
        session["summary"] = new_summary
    except Exception as exc:
        logger.warning("Session summarization failed for %s: %s", session_id, exc)
        # Re-prepend the old messages so no history is lost on failure
        session["messages"] = old_messages + session["messages"]


def build_context_for_prompt(session_id: str) -> str:
    session = get_or_create_session(session_id)
    parts: list[str] = []

    if session["summary"]:
        parts.append(f"Previous conversation summary:\n{session['summary']}")

    if session["messages"]:
        recent = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in session["messages"]
        )
        parts.append(f"Recent conversation:\n{recent}")

    return "\n\n".join(parts)


def clear_session(session_id: str) -> None:
    sessions.pop(session_id, None)
    try:
        from app.engines.analytical import clear_active_dataframe
        clear_active_dataframe(session_id)
    except Exception:
        pass
