from typing import TypedDict, Optional

class AgentState(TypedDict):
    """
    This is the shared memory of the entire graph.
    Every node reads from and writes to this state.
    Think of it as a baton passed between runners in a relay race.
    """
    question: str              # The user's original question
    session_id: str            # For conversation memory
    file_bytes: Optional[bytes]  # Uploaded file content if any
    filename: Optional[str]    # Name of uploaded file
    engine: Optional[str]      # Which engine was selected: "csv" or "pdf"
    answer: Optional[str]      # Final answer
    sources: Optional[list]    # Source documents used
    status: Optional[str]      # "success" or "error"
    error: Optional[str]       # Error message if any