from pydantic import BaseModel, Field


class Critique(BaseModel):
    """
    Represents the critique of a task attempt, including reasoning, success status, and suggestions for improvement.

    Attributes:
        reasoning (str): Explanation of why the task was succeeded or failed.
        success (bool): Indicates whether the task requirements were met.
        critique (str): Suggestions or critique to help improve future task attempts.
    """
    reasoning: str = Field(description="reason why I have succeeded or failed at the task")
    success: bool = Field(description="evaluate if I have met the task requirements")
    critique: str = Field(description="critique to help me improve")
