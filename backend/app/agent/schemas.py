# backend/app/agent/schemas.py
from pydantic import BaseModel, Field
from typing import Literal

class AssistantResponse(BaseModel):
    direct_answer: str = Field(description="The primary, direct answer to the user's question, formatted in Markdown.")
    explanation: str = Field(description="A brief 1-2 sentence explanation of how the answer was derived or what it means.")
    follow_up_hook: str = Field(description="A single, short, snappy follow-up question prompting the user's next logical step (e.g., 'Do you want me to calculate the churn rate for this plan?').")

class RouterDecision(BaseModel):
    is_clear: bool = Field(description="True if the user has a specific question. False if the request is vague like 'analyze this' or 'what do you think'.")
    engine: Literal["csv", "pdf", "clarify"] = Field(description="If is_clear is True, choose 'csv' or 'pdf'. If False, choose 'clarify'.")
    clarification_message: str = Field(description="If is_clear is False, write a friendly question asking what they want to do (e.g., 'I see your data. Do you want a summary or anomaly check?'). If clear, leave empty.")
