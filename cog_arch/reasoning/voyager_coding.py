"""
coding module as part of reasoning

heavily adapted from Voyager: An Open-Ended Embodied Agent with Large Language Models by
Wang et. al. (2023) https://github.com/MineDojo/Voyager
"""
import logging

from cognitive_base.reasoning.base_lm_reasoning import BaseLMReasoning

from ..prompts import voyager_coding_prompts as vc

from cognitive_base.utils import truncate_str
from cognitive_base.knowledge_sources.parsers import extract_blocks

from ...utils import format_voyager_progs
from ...utils.code_parse import extract_from_ast, assert_modules_in_whitelist

logger = logging.getLogger("logger")


class VoyagerCodingModule(BaseLMReasoning):
    """
    A module designed to integrate coding capabilities with a language model, facilitating
    the generation, parsing, and evaluation of code based on given tasks and feedback.
    This module is part of a cognitive architecture that aims to enable more sophisticated
    reasoning and problem-solving abilities in AI systems.

    Attributes:
        gt_fn_name: The ground truth function name expected in the generated code.
        task_prompt: The prompt describing the coding task.
        verbose: A boolean indicating if verbose logging is enabled.
        debug_mode: A boolean indicating if the module is in debug mode.
        check_imports: A boolean indicating if imports should be checked in the parsed code.
        assert_fns: A boolean indicating if the presence of functions should be asserted in the parsed code.
        rebuild_code_from_ast: A boolean indicating if the code should be rebuilt from the AST.
        sys_prompt: The system prompt template for generating system messages.
        response_template: The template for formatting responses.

    Methods:
        __init__(self, model_name, temperature, request_timeout, verbose, callbacks, debug_mode, **kwargs):
            Initializes the VoyagerCodingModule with specified parameters and model configurations.
        reset(self, full_task):
            Resets the module's state based on a new task.
        render_coding_sys_msg(self, retrieved):
            Generates a system message based on retrieved information and the system prompt template.
        render_coding_human_msg(self, env_feedback, code, critique, context, **kwargs):
            Generates a human-like message based on feedback, code, critique, and context.
        parse_ai_code(self, message):
            Parses AI-generated code from a message, extracting functions, imports, and dependencies.
        gen_code(self, retrieved, prev_code, env_out, critic_out, context, parse_tries, return_messages):
            Generates code based on retrieved information, previous code, environment output, critique, and context.
    """
    def __init__(
            self,
            model_name="gpt-3.5-turbo",
            temperature=0,
            request_timeout=120,
            verbose=True,
            callbacks=None,
            debug_mode=False,
            name='coding',
            generic_code_env=False,
            **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            verbose=verbose,
            callbacks=callbacks,
            debug_mode=debug_mode,
            name=name,
            **kwargs,
        )

        self.gt_fn_name = ''
        self.task_prompt = ''

        self.sys_prompt = vc.coding_sys_prompt
        self.response_template = vc.generic_response_template if generic_code_env else vc.fn_response_template

        self.check_imports = not generic_code_env
        self.assert_fns = not generic_code_env
        self.rebuild_code_from_ast = not generic_code_env

    """
    helper fns
    """

    def reset(self, full_task):
        self.gt_fn_name = full_task.get('gt_fn_name', '')
        self.task_prompt = full_task.get('task_prompt', full_task['task'])
        logger.info(f'The task prompt is {truncate_str(self.task_prompt)}\n')

    def render_coding_human_msg(self, env_feedback="", code="", critique="", context="", **kwargs):
        """
        Generates message based on feedback, code, critique, and context.

        Parameters:
            env_feedback (str): Feedback from the environment.
            code (str): The code generated in the last round.
            critique (str): Critique on the previous code.
            context (str): Additional context for the message.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The generated human-like message.
        """
        observation = ""
        observation += f"[Environment feedback]\n{env_feedback if env_feedback else 'None'}\n\n"
        observation += f"[Code from the last round]\n{code if code else 'No code in the first round'}\n\n"
        observation += f"[Task]\n{self.task_prompt}\n\n"
        observation += f"[Context]\n{context if context else 'None'}\n\n"
        observation += f"[Critique]\n{critique if critique else 'None'}\n\n"
        return observation

    def validate_code(self, imported_modules, functions, main_fns, fn_name):
        """
        Validates the generated code by checking imports, functions, and main function name.

        Args:
            imported_modules (list): List of imported modules.
            functions (list): List of functions in the code.
            main_fns (list): List of main functions in the code.
            fn_name (str): The name of the main function.

        Returns:
            bool: True if no parent function is found, False otherwise.
        """
        if self.check_imports:
            assert_modules_in_whitelist(imported_modules)
        if self.assert_fns:
            assert functions, 'Error: No functions found. please try again\n'

        gt_err_msg = (f"expected main function name {self.gt_fn_name} but got function name {fn_name}, try again. "
                      "Your response should declare helper functions first, then the main function last.\n")
        
        no_parent = False
        if self.gt_fn_name:
            if self.assert_fns:
                if not main_fns:
                    raise Exception("could not find any main functions (those without parent)")
                if self.gt_fn_name != fn_name:
                    raise Exception(gt_err_msg)
                no_parent = True
            else:
                no_parent = find_gt_fn(functions, self.gt_fn_name)
        return no_parent

    def parse_ai_code(self, message):
        """
        Parses AI-generated code from a message, extracting functions, imports, and dependencies.

        Parameters:
            message (AIMessage): The message containing the AI-generated code.

        Returns:
            dict: A dictionary containing parsed code information, including program code, program name, dependencies,
            and more.
        """
        code = extract_blocks(message.content, identifier='python|py')
        assert code, 'regex fails to extract Python code. check your formatting and try again\n'

        functions, import_statements, dependencies, imported_modules = extract_from_ast(code)

        main_fns = [fn for fn in functions if fn["no_parent"]]
        # main_fns can be blank if the fns are within a class Solution, so main_fns[-1] gives error
        fn_name = ''
        if main_fns:
            fn_name = main_fns[-1]['name']

        no_parent = self.validate_code(imported_modules, functions, main_fns, fn_name)

        if self.rebuild_code_from_ast:
            program_code = "\n".join(import_statements) + "\n\n".join(fn["body"] for fn in main_fns)
        else:
            program_code = code

        parsed_result = {
            "program_code": program_code,
            "program_name": fn_name,
            "dependencies": list(dependencies),
            "raw_msg": message.content,
            "no_parent": no_parent,
        }
        if self.verbose:
            for k, v in parsed_result.items():
                logger.info(f'{k}:\n {v}\n')
        return parsed_result

    """
    Reasoning Actions (from and to working mem)
    """

    def gen_code(
            self,
            retrieved,
            prev_code,
            obs,
            critic_out,
            context,
            parse_tries=3,
            return_messages=False
    ):
        """
        Generates code based on retrieved information, previous code, environment output, critique, and context.

        Parameters:
            retrieved (dict): Information retrieved from the environment or previous interactions.
            prev_code (str): The code generated in the previous round.
            critic_out (dict): Output from the critic.
            context (str): Additional context for code generation.
            parse_tries (int): The number of attempts to parse the generated code.
            return_messages (bool): If True, returns the messages used in the generation process.

        Returns:
            dict: A dictionary containing the parsed result of the generated code.
        """
        sys_vars = {
            "programs": format_voyager_progs(retrieved.get('skills', [])),
            "extra": retrieved.get('extra', ''),
            "response_format": self.response_template
        }

        human_msg = self.render_coding_human_msg(obs, prev_code, critic_out['critique'], context, **retrieved)

        parsed_result = self.lm_reason(
            sys_template=self.sys_prompt,
            human_template=human_msg,
            parse_fn=self.parse_ai_code,
            sys_vars=sys_vars,
            parse_tries=parse_tries,
            return_messages=return_messages,
        )
        return parsed_result


def find_gt_fn(functions, gt_fn_name):
    """
    Finds the ground truth function in the list of functions.

    Args:
        functions (list): List of functions in the code.
        gt_fn_name (str): The name of the ground truth function.

    Returns:
        bool: True if the ground truth function has no parent, False otherwise.

    Raises:
        Exception: If the ground truth function is not found.
    """
    for fn in functions:
        if fn['name'] == gt_fn_name:
            return fn["no_parent"]
    raise Exception(f"could not find any function with the required name {gt_fn_name}")
