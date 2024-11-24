"""
This module provides logging utilities for handling rollouts and errors.
Functions:
    log_rollout(result_dir, train, task_id, obs, info, parsed_result, critic_out):
        Logs the rollout results and saves them to a JSON file.
    handle_rollout_error(e, task_id, result_dir):
        Handles errors during rollouts by logging the error and saving the traceback to a file.
"""
import logging
import traceback
from pathlib import Path

from cognitive_base.utils import dump_json
from cognitive_base.utils.log import construct_task_folder

logger = logging.getLogger("logger")


def log_rollout(result_dir, train, task_id, obs, info, parsed_result, critic_out):
    """
    Logs the rollout results of a task.

    Args:
        result_dir (str): The directory where the results should be saved.
        train (bool): Indicates whether the mode is training or testing.
        task_id (str): The identifier of the task.
        obs (str): The environment feedback.
        info (dict): Additional information about the task.
        parsed_result (dict): The parsed result containing program code and full code.
        critic_out (dict): The output from the critic, including success status.

    Returns:
        None
    """
    mode = 'train' if train else 'test'
    output_d = {
        'env_feedback': obs,
        'state': info.get('individial_results', None),
        'code': parsed_result.get('program_code', ''),
        'full_code': parsed_result.get('full_code', ''),
        'task_id': task_id
    }
    output_d.update(critic_out)
    logger.info(f"[task_id]: {task_id} [Result]: {critic_out['success']}")
    # dump_json(output_d, f"{result_dir}/{mode}_outputs/{str(task_id).replace('/', '_')}.json")
    task_folder = construct_task_folder(result_dir, mode, task_id)
    dump_json(output_d, f"{task_folder}/output.json", indent=4)


def handle_rollout_error(e, task_id, result_dir):
    """
    Handles errors that occur during a rollout process by logging the error and writing it to a file.

    Args:
        e (Exception): The exception that was raised.
        task_id (str): The identifier of the task during which the error occurred.
        result_dir (str or Path): The directory where the error log file should be saved.

    Logs:
        Logs the error message and traceback using the logger.

    Writes:
        Writes the error message and traceback to a file named "error.txt" in the specified result directory.
    """
    unhandled_error_str = f"[task_id]: {task_id} [Unhandled Error] {repr(e)}\n"
    error_trace = traceback.format_exc()
    logger.error(f"error in rollout.\n{unhandled_error_str}\n{error_trace}")
    with open(Path(result_dir) / "error.txt", "a") as f:
        f.write(unhandled_error_str)
        f.write(error_trace)
