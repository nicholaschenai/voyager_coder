"""
stores procedures which implement actions (grounding, reasoning, learning, retrieval) or decision making
eg skill library

This implementation is heavily adapted from Voyager: An Open-Ended Embodied Agent with Large Language Models by
Wang et. al. (2023) https://github.com/MineDojo/Voyager
"""
import logging

from .base_vector_mem import BaseVectorMem

from cognitive_base.utils import f_mkdir, dump_text

logger = logging.getLogger("logger")


class VoyagerProceduralMem(BaseVectorMem):
    def __init__(
            self,
            retrieval_top_k=5,
            ckpt_dir="ckpt",
            vectordb_name="skill",
            resume=False,
            no_skill_files=False,
            **kwargs,
    ):
        super(VoyagerProceduralMem, self).__init__(
            retrieval_top_k=retrieval_top_k,
            ckpt_dir=ckpt_dir,
            vectordb_name=vectordb_name,
            resume=resume,
            **kwargs,
        )
        self.no_skill_files = no_skill_files
        if not no_skill_files:
            f_mkdir(f"{ckpt_dir}/{vectordb_name}/description")
            f_mkdir(f"{ckpt_dir}/skill/code")

        assert self.vectordb._collection.count() == len(self.fn_str_map), (
            f"Skill Manager's vectordb is not synced with entries.json.\n"
            f"There are {self.vectordb._collection.count()} skills in vectordb but \n"
            f"{len(self.fn_str_map)} skills in entries.json.\n"
            f"Did you set resume=False when initializing the manager?\n"
            f"You may need to manually delete the vectordb directory for running from scratch."
        )

    """
    helper fns
    """

    """
    Retrieval Actions (to working mem / decision procedure)
    """

    """
    Learning Actions (from working mem)
    """
    def add_skill(self, parsed_result, skill_description, task, metadata_map=None):
        """
        Adds a new skill (code) to the database.

        Args:
            parsed_result (dict): The parsed result containing skill information.
            skill_description (str): A description of the skill.
            task (str): The task associated with the skill.
            metadata_map (list, optional): A list of tuples mapping source keys to destination keys. Defaults to None.

        Returns:
            str: The name of the added skill.
        """
        result_assertion = "program_code" in parsed_result and "program_name" in parsed_result
        assert result_assertion, "program, program_name must be returned"

        program_code = parsed_result["program_code"]

        parsed_result["task"] = task

        mapping = [
            ("program_code", "code"),
            ("program_name", "name"),
            ("dependencies", "dependencies"),
            ("task", "task")
        ]
        if metadata_map:
            mapping += metadata_map
        dumped_program_name = self.add_code(parsed_result, mapping, skill_description)

        if not self.no_skill_files:
            dump_text(program_code, f"{self.ckpt_dir}/skill/code/{dumped_program_name}.py")
            dump_text(skill_description, f"{self.ckpt_dir}/skill/description/{dumped_program_name}.txt")
        self.vectordb.persist()
