"""
base curriculum module. some parts adapted from Voyager: An Open-Ended Embodied Agent with Large Language Models by
Wang et. al. (2023) https://github.com/MineDojo/Voyager
"""
import logging

from langchain.vectorstores import Chroma

from pprint import pp

from cognitive_base.reasoning.base_lm_reasoning import BaseLMReasoning

from ..prompts.answer_questions import answer_questions

from cognitive_base.utils import f_mkdir, dump_json, load_json, truncate_str
from cognitive_base.utils.llm import construct_chat_model, get_embedding_fn

logger = logging.getLogger("logger")


class BaseCurriculumModule(BaseLMReasoning):
    def __init__(
            self,
            model_name="gpt-3.5-turbo",
            curriculum_temperature=0.1,
            qa_model_name="gpt-3.5-turbo",
            qa_temperature=0,
            request_timeout=120,
            verbose=True,
            callbacks=None,
            debug_mode=False,
            ckpt_dir="ckpt",
            resume=False,
            name='curriculum',
            **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            temperature=curriculum_temperature,
            request_timeout=request_timeout,
            verbose=verbose,
            callbacks=callbacks,
            debug_mode=debug_mode,
            name=name,
            **kwargs,
        )

        self.qa_llm = construct_chat_model(
            model_name=qa_model_name,
            temperature=qa_temperature,
            request_timeout=request_timeout,
            verbose=verbose,
            callbacks=callbacks,
        )

        self.ckpt_dir = ckpt_dir

        self.completed_tasks = []
        self.failed_tasks = []
        self.qa_cache = {}

        # vectordb for qa cache
        embedding_fn = get_embedding_fn()
        f_mkdir(f"{ckpt_dir}/curriculum/vectordb")
        self.qa_cache_questions_vectordb = Chroma(
            collection_name="qa_cache_questions_vectordb",
            embedding_function=embedding_fn,
            persist_directory=f"{ckpt_dir}/curriculum/vectordb",
        )

        if resume:
            self.load_prev(ckpt_dir)
        if verbose:
            print('qa LM\n')
            pp(self.qa_llm.dict())

        logger.info(('vectordb qa_cache_questions_vectordb doc count: '
                     f'{self.qa_cache_questions_vectordb._collection.count()}\n'))

    """
    helper fns
    """

    def load_prev(self, ckpt_dir):
        """
        Loads previously completed and failed tasks from the checkpoint directory.

        Args:
            ckpt_dir (str): The directory from which to load the checkpoint data.
        """
        print(f"\033[35mLoading from {ckpt_dir}/curriculum\033[0m")
        self.completed_tasks = load_json(f"{ckpt_dir}/curriculum/completed_tasks.json").get("tasks", [])
        self.failed_tasks = load_json(f"{ckpt_dir}/curriculum/failed_tasks.json").get("tasks", [])
        self.qa_cache = load_json(f"{ckpt_dir}/curriculum/qa_cache.json")

        assert self.qa_cache_questions_vectordb._collection.count() == len(self.qa_cache), (
            f" qa cache question vectordb is not synced with qa_cache.json.\n"
            f"There are {self.qa_cache_questions_vectordb._collection.count()} questions in vectordb "
            f"but {len(self.qa_cache)} questions in qa_cache.json.\n"
            f"Did you set resume=False when initializing the agent?\n"
            f"You may need to manually delete the qa cache question vectordb directory for running from scratch.\n"
        )

    def clean_up_tasks(self):
        """
        Cleans up the lists of completed and failed tasks, removing duplicates and ensuring consistency.
        """
        updated_completed_tasks = []
        # record repeated failed tasks
        updated_failed_tasks = self.failed_tasks
        # dedup but keep order
        for task in self.completed_tasks:
            if task not in updated_completed_tasks:
                updated_completed_tasks.append(task)

        # remove completed tasks from failed tasks
        for task in updated_completed_tasks:
            while task in updated_failed_tasks:
                updated_failed_tasks.remove(task)

        self.completed_tasks = updated_completed_tasks
        self.failed_tasks = updated_failed_tasks

        # dump to json
        dump_json({"tasks": self.completed_tasks}, f"{self.ckpt_dir}/curriculum/completed_tasks.json", indent=4)
        dump_json({"tasks": self.failed_tasks}, f"{self.ckpt_dir}/curriculum/failed_tasks.json", indent=4)

        if self.verbose:
            print("\033[35m Completed tasks: " + '\n'.join([truncate_str(task) for task in self.completed_tasks]) + "\033[0m")
            print("\033[35m Failed tasks: " + '\n'.join([truncate_str(task) for task in self.failed_tasks]) + "\033[0m")

    def update_exploration_progress(self, task, success, **kwargs):
        """
        Updates the progress of task exploration.

        Args:
            task (str): The task being explored.
            success (bool): Whether the task was successfully completed.
            **kwargs: Additional keyword arguments.
        """
        if success:
            print(f"\033[35mCompleted task {truncate_str(task)}.\n\033[0m")
            self.completed_tasks.append(task)
        else:
            print(f"\033[35mFailed to complete task {truncate_str(task)}. \n Skipping to next task.\n\033[0m")
            self.failed_tasks.append(task)
        logger.info(f'Num Completed Tasks: {len(self.completed_tasks)}, Num Failed Tasks: {len(self.failed_tasks)}')
        # clean up tasks and dump to disk
        self.clean_up_tasks()

    """
    Reasoning Actions (from and to working mem)
    """
    def run_answer_questions(self, question):
        """
        Runs the question-answering process for a given question.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The answer to the question.
        """
        qa_answer = self.lm_reason(
            sys_template=answer_questions,
            human_template=f"Question: {question}",
            llm=self.qa_llm,
        )
        return qa_answer

    def get_task_context(self, task):
        """
        Constructs the context for a given task.

        Args:
            task (str): The task for which to retrieve the context.

        Returns:
            str: The context for the task.
        """
        question = f"Explain at a conceptual level, how to accomplish the below task in Python programming?\n{task}"
        if question in self.qa_cache:
            answer = self.qa_cache[question]
        else:
            answer = self.run_answer_questions(question=question)
            self.qa_cache[question] = answer
            self.qa_cache_questions_vectordb.add_texts(texts=[question])
            dump_json(self.qa_cache, f"{self.ckpt_dir}/curriculum/qa_cache.json", indent=4)
            self.qa_cache_questions_vectordb.persist()

        return f"Rough plan to accomplish the task (can be wrong): \n{answer}\n"
