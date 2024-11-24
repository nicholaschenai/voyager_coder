import argparse

from agent_expt_suite.eval_setup.argparsers import get_base_agent_parser


def add_voyager_args(parser):
    # saving
    parser.add_argument("--no_skill_files", action="store_true", help="dont save skill desc, code as separate files")

    # LM params
    parser.add_argument("--qa_model_name", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--desc_model_name", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--curriculum_temperature", type=float, default=0.1)


def get_args() -> argparse.Namespace:
    parser = get_base_agent_parser()
    add_voyager_args(parser)
    return parser.parse_args()
