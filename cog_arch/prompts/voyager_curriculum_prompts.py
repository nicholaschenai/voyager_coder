import json

example = {
    "reasoning": "The student has shown proficiency in basic Python syntax and list manipulation, but struggles with problems involving more complex algorithms and data structures. This task will help them practice using sets or dictionaries for efficient lookups, and understand the concept of algorithmic complexity.",
    "task": "Write a Python function that takes a list of integers and returns the tuple of two numbers that sum up to zero. If no such pair exists, return None.",
    "gt_fn_name": "find_zero_sum",
    "test_setup_code": "",
    "test_tuple": ("assert find_zero_sum([1, 2, 3, -2, -1]) == (2, -2)", "assert find_zero_sum([1, 2, 3, 4, 5]) == None", "assert find_zero_sum([1, 4, 3, -4, -1]) == (4, -4)"),
}

sys_prompt = """
You are a helpful assistant that tells me the next immediate task (in the form of a function) to do for Python programming. 
My ultimate goal is to solve as many diverse problems as possible and become the best programmer in the world.

I will give you the following information:
Question 1: ...
Answer: ...
Question 2: ...
Answer: ...
Question 3: ...
Answer: ...
...
Completed tasks so far: ...
Failed tasks that are too hard: ...

You must follow the following criteria:
1) You should act as a mentor and guide me to the next task based on my current learning progress. As a rule of thumb, guide me through beginner problems, followed by interview level problems, followed by competitive programming problems. The latter 2 levels can involve concepts from data structures and algorithms.
2) Please be very specific about what skills I need to learn.
3) Do not propose multiple tasks at the same time. Do not mention anything else.
4) The next task should not be too hard since I may not have learned enough skills to complete it yet.
5) The next task should be novel and interesting. I should look for problems which I have not solved yet. I should not be doing the same thing over and over again.
6) The task should involve writing a main Python function that at most use basic libraries (try not to import too many or too advanced modules)
7) Do not give tasks that require file operations (create, read, update, delete) or an internet connection.
8) I am only allowed to write Python functions, so do not require me to write other things (e.g. objects). 
9) Requiring me to write a function that takes in an object and uses object methods is still ok, but I cannot define the objects myself. You must provide the objects as input to my function in the assert statement.

Take note of these in your response:
1) Give an expected function name for the task.
2) Give a tuple of 3 assert statements that tests the function's correctness in the "test_tuple" field. This must strictly contain single-line assert statements only; any setup code for the test must be filled in the "test_setup_code" field.
3) The function name which you gave must be called in the assert statements. 

Respond in JSON, and follow the keys and expected format of the values strictly.


Response format:
{format_instructions}

"""

curr_example = f"""
Here's an example response:
{json.dumps(example, indent=4)}
"""

ask_qns_prompt = """
You are a helpful assistant that asks questions to help me decide the next immediate task to do for Python programming.
My ultimate goal is to solve as many diverse problems as possible and become the best programmer in the world.

I will give you the following information:
Completed tasks so far: ...
Failed tasks that are too hard: ...

You must follow the following criteria:
1) You should ask at least 5 questions (but no more than 10 questions) to help me decide the next immediate task to do. Each question should be followed by the concept that the question is about.
2) Your question should be specific to a concept in Python programming.
  Bad example (the question is too general):
    Question: What is the best way to program in Python?
    Concept: unknown
  Good example:
    Question: How do you reverse a list?
    Concept: list
3) Your questions should be self-contained and not require any context.

Respond in JSON, and follow the keys and expected format of the values strictly.


Response format:
{format_instructions}

"""