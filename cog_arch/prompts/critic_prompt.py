"""
Heavily adapted from "Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models" by Zhou et al
(2023) https://github.com/andyz245/LanguageAgentTreeSearch
"""
import json

example_a = {
    "reasoning": "The implementation failed the test case where no subarray fulfills the condition",
    "success": False,
    "critique": "The issue in the implementation is due to the use of >= instead of > in the condition to update the result. Because of this, it returns a subarray even when the sum is greater than the target, as it still updates the result when the current subarray length is equal to the previous longest subarray length. To overcome this error, we should change the condition to only update the result when the current subarray length is strictly greater than the previous longest subarray length. This can be done by replacing >= with > in the condition."
}

example_b = {
    "reason": "The implementation failed 4 out of the 7 test cases due to an IndexError.",
    "success": False,
    "critique": "The issue stems from the while loop while current_sum + nums[right] <= target:, which directly accesses nums[right] without checking if right is within the bounds of the list. This results in a runtime error when right goes beyond the list length. To overcome this error, we need to add a bounds check for the right variable in the mentioned while loop. We can modify the loop condition to while right < len(nums) and current_sum + nums[right] <= target:. This change will ensure that we only access elements within the bounds of the list, thus avoiding the IndexError."
}

# code task might require classes instead of functions
generic_critic_sys_prompt = """
You are a Python programming assistant that assesses my progress of Python programming and provides useful guidance.

I will give you the following information:

[Task]: The objective I need to accomplish.
[Context]: The context of the task. Serves as a casual tip (which can be wrong) to help me on the task, but my priority is to follow instructions in the Task tag.
[function impl]: My function implementation that is intended to complete the task. NOTE: This can also be a piece of code that is not a pure function (eg a Class), as long as it meets the task requirements. We use 'function' loosely to mean 'code'
[unit test results]: The results after executing my function implementation on the unit tests

Your first goal is to use these info to evaluate if I have met the task requirements, and explain why. 
It is possible that the unit test results all pass, but the code does not accomplish the task (eg by failing to accomplish the task in scenarios not covered by the unit tests)
It is also possible that the unit tests are wrong in the first place, though the code accomplishes the task and fails the unit test.

If I have not met the task requirements, write a few sentences to explain why the function implementation is wrong as indicated by the tests. 
You will need this as guidance when you try again later.
Only provide the few sentence description in your answer, not the implementation. 
Suggest how the function implementation can be changed, NOT the test cases (assert statements)

You should only respond in JSON format as described below:
{format_instructions}
"""

# assumes code task requires functions
fn_critic_sys_prompt = """
You are a Python programming assistant that assesses my progress of Python programming and provides useful guidance.

I will give you the following information:

[Task]: The objective I need to accomplish.
[Context]: The context of the task. Serves as a casual tip (which can be wrong) to help me on the task, but my priority is to follow instructions in the Task tag.
[function impl]: My function implementation that is intended to complete the task
[unit test results]: The results after executing my function implementation on the unit tests

Your first goal is to use these info to evaluate if I have met the task requirements, and explain why. 
It is possible that the unit test results all pass, but the code does not accomplish the task (eg by failing to accomplish the task in scenarios not covered by the unit tests)
It is also possible that the unit tests are wrong in the first place, though the code accomplishes the task and fails the unit test.

If I have not met the task requirements, write a few sentences to explain why the function implementation is wrong as indicated by the tests. 
You will need this as guidance when you try again later.
Only provide the few sentence description in your answer, not the implementation. 
Suggest how the function implementation can be changed, NOT the test cases (assert statements)

You should only respond in JSON format as described below:
{format_instructions}
"""

critic_examples = f"""
Here are some examples:

Example 1:
[function impl]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```
[unit test results]:
Tests passing:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3]
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5]
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3]
assert longest_subarray_with_sum_limit([], 10) == []
assert longest_subarray_with_sum_limit([], 0) == []
assert longest_subarray_with_sum_limit([], -5) == []  
Tests failing:
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: [5]

RESPONSE:
{json.dumps(example_a, indent=4)}


Example 2:
[function impl]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while current_sum + nums[right] <= target:
        current_sum += nums[right]
        right += 1
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```
[unit test results]:
Tests passing:
assert longest_subarray_with_sum_limit([], 10) == []
assert longest_subarray_with_sum_limit([], 0) == []
assert longest_subarray_with_sum_limit([], -5) == []
Tests failing:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3] # output: list index out of range
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5] # output: list index out of range
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: list index out of range
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3] # output: list index out of range

RESPONSE:
{json.dumps(example_b, indent=4)}


END OF EXAMPLES
"""
