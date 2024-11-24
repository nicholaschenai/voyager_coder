from typing import List, Tuple

# from langchain_core.pydantic_v1 import BaseModel, Field  # thing that i recently always use
from pydantic import BaseModel, Field  # original that worked


class QuestionConcept(BaseModel):
    question: str = Field(description=(
        "A question related to Python programming, to help me brainstorm the next immediate task to do"
    ))
    concept: str = Field(description="The Python programming concept associated with the question above")


# somehow List[Tuple[str, str]] gives error when using structured output so we use this
class BrainstormQns(BaseModel):
    reasoning: str = Field(description="Reason out, step-by-step, why you picked these questions and concepts")
    question_concept_list: List[QuestionConcept] = Field(description="list of questions and their associated concepts")


class NextTask(BaseModel):
    reasoning: str = Field(
        description="Based on the information listed above, reason about what the next task should be."
    )
    task: str = Field(description=(
        "The next task, which requires me to write a single function. "
        "Be sure to state the expected data structure for inputs and outputs."
    ))
    gt_fn_name: str = Field(description="The expected name for the single function for the task")
    test_setup_code: str = Field(description=(
        "setup code, if required. This will be run just before the assert statements are executed."
    ))
    test_tuple: Tuple[str, str, str] = Field(description=(
        "Tuple of 3 SINGLE-LINE test cases that starts with 'assert' to test the function's correctness."
        "Any extra required code should be written in the test_setup_code field."
    ), items={"type": "string"})
