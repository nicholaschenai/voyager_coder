"""
critic module as part of reasoning

heavily adapted from Voyager: An Open-Ended Embodied Agent with Large Language Models by
Wang et. al. (2023) https://github.com/MineDojo/Voyager
"""
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from ..prompts.critic_prompt import fn_critic_sys_prompt, generic_critic_sys_prompt, critic_examples
from .critic_pydantic_models import Critique

from cognitive_base.reasoning.base_lm_reasoning import BaseLMReasoning


class CriticModule(BaseLMReasoning):
    """
    A module responsible for evaluating the success of tasks and providing critiques using a language model 

    This module integrates with a language model to assess task outputs, offering critiques and success evaluations
    based on the generated output and environmental feedback.

    Attributes:
        use_critic (bool): Determines whether to use the critic functionality.
        use_critic_success (bool): Determines whether to use the critic for success evaluation.
        sys_prompt (str): The system prompt used for generating critiques.
    """
    def __init__(
            self,
            model_name="gpt-3.5-turbo",
            temperature=0,
            request_timeout=120,
            verbose=True,
            callbacks=None,
            debug_mode=False,
            name='critic',
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

        self.use_critic = True
        self.use_critic_success = True

        self.sys_prompt = generic_critic_sys_prompt if kwargs['generic_code_env'] else fn_critic_sys_prompt
        self.task = None
        
    """
    helper fns
    """

    @staticmethod
    def render_critic_message(generated_code, env_feedback, task, context):
        """
        Formats a message for the critic based on task details and generated code.

        Args:
            generated_code (str): The code generated as a solution to the task.
            env_feedback (str): Feedback from the environment or unit tests.
            task (str): The original task description.
            context (str): Additional context relevant to the task.

        Returns:
            HumanMessage: A message formatted for human understanding, encapsulating task details and feedback.
        """
        obs = ''
        obs += '[Task]:\n'
        obs += task + '\n'
        obs += f"[Context]: \n {context}\n\n" if context else "[Context]: None\n\n"
        obs += '[function impl]:\n'
        obs += generated_code + '\n'
        obs += '[unit test results]:\n'
        obs += env_feedback + '\n'
        return obs

    """
    Reasoning Actions (from and to working mem)
    """
    def check_success(
        self,
        env_feedback,
        reward,
        generated_code,
        context,
        max_retries=5,
    ):
        """
        Main method to check the success of a task, integrating environmental outputs and generated code.

        This method orchestrates the creation of messages for the critic, invokes the language model, and determines
        the success of the task based on the model's critique.

        Args:
            env_feedback (str): Feedback from the environment
            reward (bool): The reward status of the task
            generated_code (str): The code generated as a solution to the task.
            context (str): Additional context relevant to the task.
            max_retries (int): The maximum number of retries for language model invocation.

        Returns:
            dict: A dictionary containing the success status, critique, and reasoning.
        """
        critic_out = {'success': reward, 'critique': "", 'reasoning': ""}
        if (not reward and self.use_critic) or self.use_critic_success:
            messages_threads = self.construct_messages(
                sys_template=self.sys_prompt,
                human_template=self.render_critic_message(generated_code, env_feedback, self.task, context),
                parser=PydanticOutputParser(pydantic_object=Critique),
            )
            messages = messages_threads[0]
            messages.insert(1, SystemMessage(content=critic_examples))
            critic_out = self.lm_reason(
                messages=messages,
                structured=True,
                pydantic_model=Critique,
                fallback={'success': False, 'critique': "", 'reasoning': ""},
                parse_tries=max_retries,
            )

            if not self.use_critic_success:
                critic_out['success'] = reward

        return critic_out
