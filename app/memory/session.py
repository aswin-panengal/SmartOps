from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=settings.google_api_key,
    temperature=0
)

# In-memory session store
# Key: session_id, Value: list of messages
sessions: dict = {}

MAX_MESSAGES_BEFORE_SUMMARY = 10

def get_or_create_session(session_id: str) -> dict:
    """
    Get existing session or create a new one.
    Each session has a message history and an optional summary
    of older messages.
    """
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "summary": None
        }
    return sessions[session_id]

def add_message(session_id: str, role: str, content: str):
    """
    Add a message to the session history.
    Role is either 'user' or 'assistant'.
    """
    session = get_or_create_session(session_id)
    session["messages"].append({
        "role": role,
        "content": content
    })

    # If history is getting long, summarize older messages
    if len(session["messages"]) > MAX_MESSAGES_BEFORE_SUMMARY:
        _summarize_old_messages(session_id)

def _summarize_old_messages(session_id: str):
    """
    When conversation gets too long:
    1. Take the oldest half of messages
    2. Ask Gemini to summarize them
    3. Replace those messages with the summary
    4. Keep only recent messages in full
    
    This prevents token limits from being hit on long conversations.
    """
    session = sessions[session_id]
    messages = session["messages"]

    # Split: summarize the older half, keep the recent half
    midpoint = len(messages) // 2
    old_messages = messages[:midpoint]
    recent_messages = messages[midpoint:]

    # Build text to summarize
    conversation_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in old_messages
    ])

    summary_prompt = f"""
    Summarize this conversation history concisely.
    Preserve all important facts, decisions, and context.
    
    Conversation:
    {conversation_text}
    
    Summary:
    """

    response = llm.invoke(summary_prompt)
    new_summary = response.content

    # If there was already a previous summary, combine them
    if session["summary"]:
        new_summary = f"{session['summary']}\n\n{new_summary}"

    session["summary"] = new_summary
    session["messages"] = recent_messages

def build_context_for_prompt(session_id: str) -> str:
    """
    Build the conversation context to inject into prompts.
    Includes the summary (if any) + recent messages.
    This is what gets passed to Gemini with every new question.
    """
    session = get_or_create_session(session_id)
    context_parts = []

    if session["summary"]:
        context_parts.append(
            f"Previous conversation summary:\n{session['summary']}"
        )

    if session["messages"]:
        recent = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in session["messages"]
        ])
        context_parts.append(f"Recent conversation:\n{recent}")

    return "\n\n".join(context_parts)

def clear_session(session_id: str):
    """Reset a session completely."""
    if session_id in sessions:
        del sessions[session_id]