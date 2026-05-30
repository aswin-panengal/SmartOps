from pydantic import BaseModel, Field
from typing import List

class AssistantResponse(BaseModel):
    direct_answer: str = Field(description="The primary, direct answer to the user's question, formatted in Markdown.")
    explanation: str = Field(description="A brief 1-2 sentence explanation of how the answer was derived or what it means.")
    follow_up_questions: List[str] = Field(description="Exactly 3 suggested follow-up questions the user might want to ask next.", max_items=3, min_items=3)