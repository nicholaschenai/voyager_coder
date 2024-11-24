"""
Reimplementation of "Voyager: An Open-Ended Embodied Agent with Large Language Models" by
Wang et. al. (2023) https://github.com/MineDojo/Voyager

We load reasoning modules manually to let IDE catch missing modules
"""

import logging

from agent_expt_suite.envs import env_globals

from cognitive_base.agents.base_agent import BaseAgent

from ...utils.log import log_rollout, handle_rollout_error
from ...utils.code_parse import append_dependencies

logger = logging.getLogger("logger")


class VoyagerAgent(BaseAgent):
    def __init__(self, args, components, handler):
        super().__init__(args, components, handler)
        """
        Working Memory: short term memory reflecting current circumstances
        """
        self.last_events = None
        self.context = None

        """
        Reasoning modules
        """
        self.coding_module = components['coding_module']
        self.critic_module = components['critic_module']
        self.curriculum_module = components['curriculum_module']
        self.desc_module = components['desc_module']

    """
    helper fns
    """
    def set_context(self, full_task):
        """
        Sets the context for the current task by retrieving task-specific context from the curriculum module.
        Separated from get_next_task in original Voyager since we need context for testing
        but not task proposal meant for training

        :param full_task: A dictionary containing details of the task to be performed.
        """
        self.context = self.curriculum_module.get_task_context(full_task['task'])

    def reset(self, full_task):
        super().reset(full_task)
        self.last_events = ''
        self.coding_module.reset(full_task)
        self.critic_module.task = self.task
        self.set_context(full_task)

    def retrieve_for_coding(self, cue):
        """
        Retrieves relevant coding skills based on the given cue from the procedural memory.

        :param cue: A string or structured data used as a cue to retrieve relevant code snippets or skills.
        :return: A dictionary containing retrieved skills.
        """
        retrieved = {'skills': self.procedural_mem.retrieve_code(cue)}
        return retrieved

    def get_curriculum_inputs(self):
        """
        Retrieves inputs required for the curriculum module to choose next task.
        mainly used when inherited so that we can pass in existing skill names to prevent naming collisions
        """
        return {}

    def get_fn_maps(self):
        """
        Retrieves function mappings from the procedural memory to support code generation.

        :return: A list containing function string mappings.
        """
        return [self.procedural_mem.fn_str_map]

    def process_transition(self, transition_info):
        """
        Processes transitions between states in the environment. This method is a placeholder in the base
        implementation and can be overridden for environments where transitions need to be explicitly handled.

        :param info: A tuple containing information about the action taken, environment outputs, and critic output.
        """
        # Note: curriculum takes in events to render observation, assuming the env keeps going on
        # code env is reset everytime so no need, but in open world we will need this
        # action, env_outputs, critic_out = info
        # self.last_events = copy.deepcopy(env_outputs)
        pass

    def rollout(self, full_task, use_public_tests=False):
        """
        Performs a rollout for a given task, attempting to solve it through multiple iterations. It involves
        generating code, evaluating + receiving feedback from critic.

        :param full_task: A dictionary containing details of the task to be performed.
        :param use_public_tests: A boolean indicating whether to use public tests for evaluation.
        :return: A tuple containing a success flag and the parsed result of the last code generation attempt.
        """
        self.reset(full_task)

        obs, reward, info = '', False, {}
        parsed_result, code = {}, ''
        critic_out = {'success': False, 'critique': "", 'reasoning': ""}
        
        logger.info(f'Attempting task_id {self.task_id}')
        
        try:
            for i in range(self.max_task_attempts):
                logger.info(f"\033[35m Rollout attempt {i + 1}/{self.max_task_attempts}\033[0m")

                # Note: original Voyager uses context and chatlog summary of events (handcrafted filters) to retrieve
                retrieved = self.retrieve_for_coding(self.context + "\n\n" + critic_out['critique'])

                parsed_result = self.coding_module.gen_code(retrieved, code, obs, critic_out, self.context)
                if parsed_result:
                    code = parsed_result["program_code"]
                    full_code = append_dependencies(parsed_result, self.get_fn_maps())

                    if self.eval_later and not self.train and not use_public_tests:
                        break

                    obs, reward, _, info = env_globals.task_env.step(full_code, use_public_tests)

                    critic_out = self.critic_module.check_success(obs, reward, code, self.context)

                    transition_info = {'parsed_result': parsed_result, 'obs': obs, 'critic_out': critic_out, 'i': i}
                    self.process_transition(transition_info)

                    if critic_out['success']:
                        break
        except Exception as e:
            handle_rollout_error(e, self.task_id, self.args['result_dir'])

        log_rollout(self.args['result_dir'], self.train, self.task_id, obs, info, parsed_result, critic_out)
        return critic_out['success'], parsed_result

    """
    decision procedures
    """

    def train_step(self):
        full_task = self.curriculum_module.get_next_task(**self.get_curriculum_inputs())

        success, parsed_result = self.rollout(full_task)

        if success:
            code_desc = self.desc_module.gen_code_desc(**parsed_result, task=self.task)
            self.procedural_mem.add_skill(parsed_result, code_desc, self.task)

        self.curriculum_module.update_exploration_progress(self.task, success)

    def test_one(self, full_task):
        return self.rollout(full_task, use_public_tests=self.use_public_tests)
