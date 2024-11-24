from .code_parse import get_call_str

def assert_task_not_completed(task, completed_tasks):
    """
    Asserts that the given task has not been completed previously.

    Args:
        task (str): The task to be checked.
        completed_tasks (list): A list of tasks that have already been completed.

    Raises:
        AssertionError: If the task has already been completed.
    """
    err_msg = ("The task proposed has already been completed previously!"
               "Give a new task that has not been completed before")
    assert task not in completed_tasks, err_msg


def assert_task_not_previous(task, previous_task):
    """
    Asserts that the given task is not the same as the previous task.

    Args:
        task (str): The task to be checked.
        previous_task (str): The previous task.

    Raises:
        AssertionError: If the task is the same as the previous task.
    """
    assert task != previous_task, "The task you proposed is the previous task. Issue a different task."


def assert_no_function_name_collision(function_name, excluded_names):
    """
    Asserts that the given function name does not collide with any excluded names.

    Args:
        function_name (str): The function name to be checked.
        excluded_names (list): A list of excluded function names.

    Raises:
        AssertionError: If the function name collides with any excluded names.
    """
    if not excluded_names:
        return
    err_msg = f"gt_fn_name {function_name} is currently used by another process. please give a different gt_fn_name"
    assert function_name not in excluded_names, err_msg


def assert_single_gt_fn_name(function_name):
    """
    Asserts that the given function name (expected ground truth fn name) contains only one function.

    Args:
        function_name (str): The function name to be checked.

    Raises:
        AssertionError: If the function name contains more than one function.
    """
    num_functions = len(function_name.split(" "))
    assert num_functions == 1, f"you can only request for ONE function to be written! found {num_functions} functions"


def extract_assert_function_call(test_case):
    """
    Extracts the function call from an assert statement in the test case.

    Args:
        test_case (str): The test case containing the assert statement.

    Returns:
        str: The extracted function call.

    Raises:
        Exception: If the assert statement or function call is invalid.
    """
    try:
        func_call = get_call_str(test_case)
        return func_call
    except Exception as e:
        raise Exception(f"Invalid assert statement/function call: {test_case}\n{str(e)}, {type(e).__name__}\n")
    
def assert_test_case_has_assert(test_case):
    """
    Asserts that the test case starts with an assert statement.

    Args:
        test_case (str): The test case to be checked.

    Raises:
        AssertionError: If the test case does not start with an assert statement.
    """
    err_msg = (f"this test case does not start with assert: {test_case}\n"
               "Make sure that each test case starts with assert and only contains one line")
    assert test_case.startswith('assert'), err_msg


def assert_function_name_in_call(gt_fn_name, func_call):
    """
    Asserts that the function call contains the ground truth function name.

    Args:
        gt_fn_name (str): The ground truth function name.
        func_call (str): The function call to be checked.

    Raises:
        AssertionError: If the function call does not contain the ground truth function name.
    """
    assert gt_fn_name in func_call, f"test case called {func_call} but it did not contain gt_fn_name {gt_fn_name}"


def perform_task_assertions(task, completed_tasks, previous_task):
    """
    Performs a series of assertions to validate the task.

    Args:
        task (str): The task to be validated.
        completed_tasks (list): A list of tasks that have already been completed.
        previous_task (str): The previous task.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    assert task, "task not found!"
    assert_task_not_completed(task, completed_tasks)
    assert_task_not_previous(task, previous_task)

def construct_valid_test_cases(raw_test_cases, gt_fn_name):
    """
    Constructs a list of valid test cases by validating each proposed test case.

    Args:
        raw_test_cases (list): A list of raw test cases.
        gt_fn_name (str): The ground truth function name.

    Returns:
        list: A list of valid test cases.

    Raises:
        Exception: If any of the test cases are invalid.
    """
    valid_test_cases = []
    for test_case in raw_test_cases:
        try:
            assert_test_case_has_assert(test_case)
            func_call = extract_assert_function_call(test_case)
            assert_function_name_in_call(gt_fn_name, func_call)
        except Exception as e:
            # deprecate this numbering thing. if LM is bad, let it be
            # if last iteration, skip appending error test cases instead of retrying a new set of test cases
            # if i == max_propose_retries - 1:
            #     continue
            raise Exception(f"{str(e)}, {type(e).__name__}\n")
        valid_test_cases.append(test_case)
    return valid_test_cases
