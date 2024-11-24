"""
curriculum module adapted from Voyager: An Open-Ended Embodied Agent with Large Language Models by
Wang et. al. (2023) https://github.com/MineDojo/Voyager
"""
import logging
from urllib import response

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser

from pprint import pp

from ..prompts import voyager_curriculum_prompts as curr_prompts

from .base_curriculum import BaseCurriculumModule
from .voyager_curriculum_pydantic_models import BrainstormQns, NextTask

from ...utils.code_parse import extract_from_ast, assert_modules_in_whitelist
from ...utils import curriculum_validation as cv

from cognitive_base.utils import custom_breakpoint, dump_json, pydantic_parse_fn

logger = logging.getLogger("logger")


class VoyagerCurriculumModule(BaseCurriculumModule):
    """
    Manages the curriculum for a learning model, handling tasks, questions, and answers.

    This module is responsible for initializing the curriculum and question-answering models, managing tasks

    according to paper, all temp 0 except curriculum which has temp 0.1 for diversity
    Attributes:
        model_name (str): Name of the model used for the curriculum.
        curriculum_temperature (float): Temperature setting for curriculum diversity.
        qa_model_name (str): Name of the model used for question answering.
        qa_temperature (float): Temperature setting for the QA model.
        request_timeout (int): Timeout for model requests.
        verbose (bool): Flag to enable verbose logging.
        callbacks (list): List of callback functions.
        max_propose_retries (int): Maximum number of retries for proposing tasks.
        debug_mode (bool): Flag to enable debug mode.
        ckpt_dir (str): Directory for checkpoints.
        resume (bool): Flag to enable resuming from checkpoints.
    """
    def __init__(
            self,
            model_name="gpt-3.5-turbo",
            curriculum_temperature=0.1,
            qa_model_name="gpt-3.5-turbo",
            qa_temperature=0,
            request_timeout=120,
            verbose=True,
            callbacks=None,
            max_propose_retries=5,
            debug_mode=False,
            ckpt_dir="ckpt",
            resume=False,
            name='curriculum',
            **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            curriculum_temperature=curriculum_temperature,
            request_timeout=request_timeout,
            verbose=verbose,
            callbacks=callbacks,
            debug_mode=debug_mode,
            name=name,
            qa_model_name=qa_model_name,
            qa_temperature=qa_temperature,
            ckpt_dir=ckpt_dir,
            resume=resume,
            **kwargs,
        )

        self.max_propose_retries = max_propose_retries
        self.previous_task = ''

    """
    helper fns
    """
    def update_exploration_progress(self, task, success, **kwargs):
        """
        Updates the exploration progress based on the task completion status.

        Args:
            task (dict): Task information.
            success (bool): Flag indicating task completion status.
            **kwargs: Additional keyword arguments.
        """
        super().update_exploration_progress(task, success, **kwargs)
        self.previous_task = task

    @staticmethod
    def render_observation(completed_tasks, failed_tasks):
        """
        Renders the observation of completed and failed tasks into a structured format.

        Args:
            completed_tasks (list): List of completed tasks.
            failed_tasks (list): List of failed tasks.

        Returns:
            dict: A dictionary containing the observation context, completed tasks, and failed tasks.
        """
        completed_tasks_str = ("\n\t" + "\n\t".join(completed_tasks)) if completed_tasks else "None"
        failed_tasks_str = ("\n\t" + "\n\t".join(failed_tasks)) if failed_tasks else "None"

        observation = {
            "context": "",
            "completed_tasks": f"Completed tasks so far (do not give me these tasks again!): {completed_tasks_str}\n\n",
            "failed_tasks": f"Failed tasks that are too hard: {failed_tasks_str}\n\n",
        }
        return observation

    @staticmethod
    def render_human_msg(questions, answers, observation):
        """
        Renders the curriculum's human-readable message based on questions, answers, and observations.

        Args:
            questions (list): List of questions.
            answers (list): List of answers.
            observation (dict): Observation data including context, completed tasks, and failed tasks.
        """
        i = 1
        for question, answer in zip(questions, answers):
            if "Answer: Unknown" in answer or "language model" in answer:
                continue
            observation["context"] += f"Question {i}: {question}\n"
            # LM already tasked to give 'Answer:' in response but that is not validated
            observation["context"] += f"{answer}\n\n"
            i += 1
            if i > 5:
                break

        content = ""
        for key in observation:
            content += observation[key]

        print(f"\033[35m****Curriculum Agent human message****\n{content}\033[0m")
        return content

    def run_qa(self, observation):
        """
        Executes a question-answering process based on the given observation.

        This method generates new questions from the observation, checks for cached answers,
        and retrieves or computes answers as necessary. It updates the cache with new questions
        and answers, and persists the cache state.

        Args:
            observation (str): The input observation from which questions are generated.

        Returns:
            tuple: A tuple containing two lists:
                - questions (list of str): The list of questions generated from the observation.
                - answers (list of str): The list of answers corresponding to the questions.
        """
        questions_new, _ = self.run_ask_questions(observation)
        questions = []
        answers = []
        for question in questions_new:
            if self.qa_cache_questions_vectordb._collection.count() > 0:
                # removed brackets () from output, if it gives error then put it back
                docs_and_scores = self.qa_cache_questions_vectordb.similarity_search_with_score(question, k=1)
                # retrieve similar Q,A pairs. metric is L2 so small is good
                if docs_and_scores and docs_and_scores[0][1] < 0.05:
                    question_cached = docs_and_scores[0][0].page_content
                    assert question_cached in self.qa_cache
                    answer_cached = self.qa_cache[question_cached]
                    questions.append(question_cached)
                    answers.append(answer_cached)
                    continue
            answer = self.run_answer_questions(question=question)
            assert question not in self.qa_cache
            self.qa_cache[question] = answer
            self.qa_cache_questions_vectordb.add_texts(texts=[question])
            dump_json(self.qa_cache, f"{self.ckpt_dir}/curriculum/qa_cache.json", indent=4)
            self.qa_cache_questions_vectordb.persist()
            questions.append(question)
            answers.append(answer)
        assert len(questions_new) == len(questions) == len(answers)
        return questions, answers

    def response_validation(self, response):
        """
        Validates the response dictionary (task info from curriculum) by performing various checks and assertions.
        Args:
            response (dict): The response dictionary to validate. It should contain the following keys:
                - 'task': A string representing the task.
                - 'test_tuple': A list representing the test tuple (string of assertion).
                - 'test_setup_code': A string containing the test setup code.
                - 'gt_fn_name': A string representing the ground truth function name.
        Raises:
            AssertionError: If any of the validation checks fail, an assertion error is raised with an appropriate message.
        Modifies:
            response (dict): Adds a 'test_list' key to the response dictionary containing valid test cases.
        """
        task = response.get('task', '')
        cv.perform_task_assertions(task, self.completed_tasks, self.previous_task)
        
        test_tuple = response.get('test_tuple', [])
        assert test_tuple, "test_tuple not found!"

        _, _, _, imported_modules = extract_from_ast(response['test_setup_code'])
        assert_modules_in_whitelist(imported_modules)

        gt_fn_name = response['gt_fn_name']
        # cv.assert_no_function_name_collision(gt_fn_name, excluded_names)
        cv.assert_single_gt_fn_name(gt_fn_name)

        valid_test_cases = cv.construct_valid_test_cases(test_tuple, gt_fn_name)
        assert valid_test_cases, "could not parse all test cases"
        response['test_list'] = valid_test_cases

    def parse_n_validate(self, message, parser):
        response = pydantic_parse_fn(message, parser)
        try:
            self.response_validation(response)
        except Exception as e:
            error_msg = (
                f"Error! {str(e)}, {type(e).__name__}\n"
                "Check your response again, and fix the error above to follow the required format.\n"
                "If the same type of error repeats, give a different task which avoids the error above.\n"
            )
            raise Exception(error_msg)
        return response
    
    """
    Reasoning Actions (from and to working mem)
    """
    def run_ask_questions(self, observation):
        """
        Generates a list of questions based on the given observation.
        This method processes the observation data, constructs a prompt, and uses a language model
        to generate a list of questions. The questions are extracted from the model's output.
        Args:
            observation (dict): A dictionary containing observation data.
        Returns:
            tuple: A tuple containing a list of generated questions and an empty list.
        """
        content = ""
        for key in observation:
            content += observation[key]

        out = self.lm_reason(
            sys_template=curr_prompts.ask_qns_prompt,
            human_template=content,
            structured=True,
            pydantic_model=BrainstormQns,
            fallback={"question_concept_list": []},
            llm=self.qa_llm,
        )

        questions = [question_concept['question'] for question_concept in out.get('question_concept_list', [])]
        return questions, []
    
    # def prompt_parse_loop(self, messages, output_parser):
    #     for i in range(self.max_propose_retries):
    #         logger.info(f'curriculum parsing attempt {i + 1} / {self.max_propose_retries}\n')
    #         ai_message = self.llm.invoke(messages)
    #         try:
    #             response = output_parser.invoke(ai_message)
    #             if self.debug_mode:
    #                 pp(response)
    #                 custom_breakpoint()
    #             self.response_validation(response)
    #             pp(response)
    #             return response
    #         except Exception as e:
    #             error_msg = (
    #                 f"Error! {str(e)}, {type(e).__name__}\n"
    #                 "Check your response again, and fix the error above to follow the required format.\n"
    #                 "If the same type of error repeats, give a different task which avoids the error above.\n"
    #             )
    #             logger.warning(error_msg)
    #             messages.extend([AIMessage(content=ai_message.content), SystemMessage(content=error_msg)])

    #     raise RuntimeError("Max retries reached, failed to propose ai task.")

    def propose_next_ai_task(self, questions, answers, observation, excluded_names=None):
        """
        Propose the next AI task based on the given questions, answers, and observation.
        Args:
            questions (list): A list of questions to consider.
            answers (list): A list of answers corresponding to the questions.
            observation (str): The current observation or context.
            excluded_names (list, optional): A list of function names to exclude from the proposal. Defaults to None.
        Returns:
            dict: A dictionary containing the proposed task details, including:
                - 'task': The proposed task description.
                - 'gt_fn_name': The proposed function name.
                - 'test_setup_code': The setup code for testing the proposed task.
                - 'test_list': A list of tests for the proposed task.
                - 'task_prompt': The task prompt with the function name to follow.
        Raises:
            RuntimeError: If the maximum number of retries is reached and no valid task is proposed.
        """
        human_msg = self.render_human_msg(questions, answers, observation)
        parser = PydanticOutputParser(pydantic_object=NextTask)
        messages_threads = self.construct_messages(curr_prompts.sys_prompt, human_msg, parser=parser)

        # sys_msg_prompt = SystemMessagePromptTemplate.from_template(curr_prompts.sys_prompt)
        # output_parser = JsonOutputParser(pydantic_object=NextTask)
        # system_message = sys_msg_prompt.format(format_instructions=output_parser.get_format_instructions())
        # system_example_message = SystemMessage(content=curr_prompts.curr_example)
        
        messages = messages_threads[0]
        messages.insert(1, SystemMessage(content=curr_prompts.curr_example))
        
        # messages = [system_message, system_example_message, human_msg]
        # response = self.prompt_parse_loop(messages, output_parser, excluded_names)

        response = self.lm_reason(
            messages=messages,
            structured=True,
            pydantic_model=NextTask,
            parse_tries=self.max_propose_retries,
            parse_fn=self.parse_n_validate,
        )

        if not response:
            raise RuntimeError("Max retries reached, failed to propose ai task.")

        gt_fn_name = response['gt_fn_name']
        if excluded_names and gt_fn_name in excluded_names:
            # Find a non-colliding function name by appending _v2, _v3, etc.
            base_name = gt_fn_name
            version = 1
            while gt_fn_name in excluded_names:
                version += 1
                gt_fn_name = f"{base_name}_v{version}"
            
            # Replace all occurrences of the original function name in the response
            fields_to_update = ['task', 'gt_fn_name', 'test_setup_code', 'test_list']
            for field in fields_to_update:
                if field == 'test_list':
                    response[field] = [test.replace(base_name, gt_fn_name) for test in response[field]]
                else:
                    response[field] = response[field].replace(base_name, gt_fn_name)

        response['task_prompt'] = f"{response['task']}\nYou must strictly follow the function name: {response['gt_fn_name']}"
        return response

    def get_next_task(self, excluded_names=None):
        """
        Determines the next task to be performed by the AI.

        This method generates an observation based on completed and failed tasks,
        runs a question-answering process on the observation, and proposes the next
        AI task while ensuring that the new task names do not conflict with existing ones.

        Args:
            excluded_names (list, optional): A list of task names to be excluded from consideration.

        Returns:
            str: The proposed next task for the AI to perform.
        """
        observation = self.render_observation(self.completed_tasks, self.failed_tasks)
        questions, answers = self.run_qa(observation)
        # included semantic mem (RAG db) here to check that new fn names dont conflict with those.
        full_task = self.propose_next_ai_task(questions, answers, observation, excluded_names)
        return full_task
