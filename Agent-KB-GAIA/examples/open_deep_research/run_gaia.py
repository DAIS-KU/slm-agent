import argparse
import json
import os
import threading
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    Union,
    Literal,
)
import sys
from collections import Counter
import logging
import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
import torch

from scripts.scorer import question_scorer
from scripts.reformulator import prepare_response
from scripts.searcher import SearchTool
from scripts.run_agents import (
    get_single_file_description,
    get_zip_description,
)
from scripts.text_inspector_tool import TextInspectorTool
from scripts.audio_inspector_tool import AudioInspectorTool
from scripts.visual_inspector_tool import VisualInspectorTool
from scripts.async_web_crawler import (
    CrawlerReadTool,
    CrawlerArchiveSearchTool,
    SimpleCrawler,
)
from scripts.automodel import (
    get_api_model,
    get_together_model,
    process_selected_tasks_param,
    prepare_model_kwargs,
)

from agent_kb.agent_kb_utils import AKBClient, call_model, SubAKBClient, BUAKBClient
from agent_kb.agent_kb_utils_ts import AKBClientTS, build_additional_knowledge

from planner_kb import (
    proposal_planning,
    plan_mode_planning,
    bu_dynamic_proposal_planning,
)

from smolagents.memory import ActionStep, PlanningStep, TaskStep
from smolagents.agents import populate_template

from tqdm import tqdm

from smolagents import (
    CodeAgent,
    Model,
    ToolCallingAgent,
    TransformersModel,
)
from dotenv import load_dotenv

load_dotenv()

AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
    "random",
    "re",
    "sys",
    "shutil",
]


parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
env_path = os.path.join(parent_dir, ".env")

load_dotenv(dotenv_path=env_path, override=True)
login(os.getenv("HF_TOKEN"))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

jsonl_lock = threading.Lock()
trajectory_lock = threading.Lock()


def load_task_dict_from_jsonl(path: str):
    task_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for record in data:
                task_dict[record["task_id"]] = record
            return task_dict
    except json.JSONDecodeError:
        pass  # JSON 배열이 아니면 아래 JSONL 모드로
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        task_dict[record["task_id"]] = record
    return task_dict


def append_dict_to_jsonl(file_path, dict_data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        json_line = json.dumps(dict_data, ensure_ascii=False)
        f.write(json_line + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--model-id-search", type=str, default="Qwen/Qwen3-4B-Instruct-2507"
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument(
        "--level",
        type=str,
        default="all",
        choices=["all", "1", "2", "3"],
    )
    parser.add_argument(
        "--selected-tasks",
        default=None,
        nargs="*",
        help="Tasks to run: specify single or multiple indices (--selected-tasks 1 or --selected-tasks 1 2 5), a single task ID, or a path to a text file with one task ID per line",
    )
    # infer params
    parser.add_argument(
        "--planning_interval", type=int, default=1, help="Number of rollouts per state."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=12,
        help="Maximum number of steps for ReAct agent.",
    )
    parser.add_argument(
        "--temperature",
        default=None,
        type=float,
        help="The temperature for llm generation.",
    )
    parser.add_argument(
        "--top_p", default=None, type=float, help="The top_p for llm generation."
    )
    parser.add_argument(
        "--search_reflection", action="store_true", help="Enable reflection"
    )
    # agent_kb params
    parser.add_argument(
        "--agent_kb", action="store_true", help="Enable knowledge base retrieval"
    )
    parser.add_argument(
        "--apply_student", action="store_true", help="Enable student correction"
    )
    parser.add_argument(
        "--apply_teacher", action="store_true", help="Enable teacher correction"
    )
    parser.add_argument("--slm", action="store_true", help="Enable SLM agent")
    parser.add_argument(
        "--retrieval_type", type=str, default="hybrid", help="search type"
    )
    parser.add_argument("--top_k", type=int, default=3, help="top_k retrieval")
    parser.add_argument(
        "--model_name_retrieval",
        type=str,
        default="gpt-4.1",
        help="agent kb model choice",
    )
    parser.add_argument(
        "--do_field",
        type=str,
        default="do_raw",
        choices=["do_raw", "do_sum", "procedure"],
    )
    parser.add_argument("--use_sub_ex", action="store_true")
    parser.add_argument(
        "--retrieval_option",
        type=str,
        default="task_text",
        choices=["task_text", "type_and_domain"],
        help="Retrieval mode for proposal_planning",
    )
    parser.add_argument(
        "--plan_mode",
        type=str,
        default="plan",
        choices=["None", "plan", "subtask", "plan_subtask", "plan_subtask_action"],
        help="'None': original planning (no KB), 'plan': KB plan only, 'subtask': KB subtasks only, 'plan_subtask': KB plan+subtasks",
    )
    parser.add_argument(
        "--kb_type",
        type=str,
        default="proposal",
        choices=["proposal", "original", "plan_mode", "bu"],
        help=(
            "KB mode: 'proposal' uses proposal_planning (default), "
            "'original' uses agent_kb_utils_ts simple injection, "
            "'plan_mode' uses static KB injection, "
            "'bu' uses bu-taxonomy filtered hybrid search + dynamic proposal (port 8006)"
        ),
    )
    parser.add_argument(
        "--bu_depth",
        type=str,
        default=None,
        choices=["plan_only", "plan_subtask", "full"],
        help="bu kb_type 전용: depth 고정 (None이면 mapping table 동적 결정)",
    )
    return parser.parse_args()


logger.warning(
    "Make sure you deactivated Tailscale VPN, else some URLs will be blocked!"
)

USE_OPEN_MODELS = False

SET = "validation"

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all", num_proc=1)[SET]
eval_ds = eval_ds.rename_columns(
    {"Question": "question", "Final answer": "true_answer", "Level": "task"}
)


def preprocess_file_paths(row):
    if len(row["file_name"]) > 0:
        row["file_name"] = f"data/gaia/{SET}/" + row["file_name"]
    return row


eval_ds = eval_ds.map(preprocess_file_paths)
eval_df = pd.DataFrame(eval_ds)

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
serp_api_key = os.getenv("SERP_API_KEY")
BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": serp_api_key,
    "num": 10,
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent_hierarchy(
    model: Model, model_search: Model, args, debug=False,
):
    manager_agent = CodeAgent(
        model=model,
        tools=[],
        max_steps=args.max_steps,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=args.planning_interval,
        managed_agents=[],
        debug=debug,
        agent_kb=args.agent_kb,
        top_k=args.top_k,
        retrieval_type=args.retrieval_type,
        plan_mode=args.plan_mode,
    )
    return manager_agent


def append_answer(entry: dict, jsonl_file: str, file_lock) -> None:
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(entry) + "\n"
    with file_lock:
        with open(jsonl_file, "a", encoding="utf-8") as fp:
            fp.write(data)
    assert os.path.exists(jsonl_file), "File not found!"
    logger.info("Answer exported to file: {}".format(jsonl_file.resolve()))


def answer_single_question(
    example,
    args,
    model_id,
    model_id_search,
    answers_file,
    debug=False,
    retrieval=False,
    apply_student=False,
    apply_teacher=False,
    slm=False,
    model=None,
    model_search=None,
    use_sub_ex=False,
):
    if slm:
        model_name, key, url, _ = get_together_model(None)
        model_name, key_search, url_search, _ = get_together_model(None)
    else:
        model_name, key, url, model_wrapper = get_api_model(model_id)
        model_name_search, key_search, url_search, model_wrapper_search = get_api_model(
            model_id_search
        )

        kwargs = prepare_model_kwargs(model_id, args)
        kwargs_search = prepare_model_kwargs(model_id_search, args)

        model = model_wrapper(
            model_name,
            custom_role_conversions=custom_role_conversions,
            max_completion_tokens=8192,
            api_key=key,
            api_base=url,
            **kwargs,
        )

        model_search = model_wrapper_search(
            model_name_search,
            custom_role_conversions=custom_role_conversions,
            max_completion_tokens=8192,
            api_key=key_search,
            api_base=url_search,
            **kwargs_search,
        )

    document_inspection_tool = TextInspectorTool(model, 100000)
    audio_inspection_tool = AudioInspectorTool(model, 100000)
    visual_inspection_tool = VisualInspectorTool(model, 100000)

    akb_client = AKBClient()
    sub_akb_client = SubAKBClient()
    agent = create_agent_hierarchy(
        model, model_search, args, debug,
    )

    model_name_retrieval = args.model_name_retrieval
    retrieval_method = {
        "hybrid": akb_client.hybrid_search,
        "text": akb_client.text_search,
        "semantic": akb_client.semantic_search,
    }[args.retrieval_type]
    augmented_question = "Here is the task:" + example["question"]

    if example["file_name"]:
        if ".zip" in example["file_name"]:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"],
                example["question"],
                visual_inspection_tool,
                document_inspection_tool,
                audio_inspection_tool,
            )
        else:
            prompt_use_files = (
                "\n\nTo solve the task above, you will have to use this attached file:"
            )
            prompt_use_files += get_single_file_description(
                example["file_name"],
                example["question"],
                visual_inspection_tool,
                document_inspection_tool,
                audio_inspection_tool,
            )
        augmented_question += prompt_use_files

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _additional_knowledge_for_planning = None
    # try:
    if retrieval:
        if args.kb_type == "original":
            akb_ts_client = AKBClientTS()
            ts_retrieval_fn = {
                "hybrid": akb_ts_client.hybrid_search,
                "text": akb_ts_client.text_search,
                "semantic": akb_ts_client.semantic_search,
            }[args.retrieval_type]
            ts_results = ts_retrieval_fn(example["question"], top_k=args.top_k)
            _additional_knowledge_for_planning = build_additional_knowledge(ts_results)
        elif args.kb_type == "plan_mode" and args.plan_mode != "None":
            _additional_knowledge_for_planning = plan_mode_planning(
                example=example,
                retrieval_method=retrieval_method,
                top_k=args.top_k,
                plan_mode=args.plan_mode,
            )
        elif args.kb_type == "proposal" and args.plan_mode != "None":
            def planner_fn(task: str, tools, managed_agents) -> dict:
                return proposal_planning(
                    example=example,
                    augmented_question=augmented_question,
                    model_name=model_name,
                    key=key,
                    url=url,
                    model=model,
                    slm=slm,
                    retrieval_method=retrieval_method,
                    top_k=3,
                    retrieval_option=args.retrieval_option,
                    type_domain_retrieval_method=akb_client.type_domain_text_search,
                    plan_mode=args.plan_mode,
                    tools=tools,
                    managed_agents=managed_agents,
                    planning_prompt_templates=agent.prompt_templates["planning"],
                )

            agent.planner_fn = planner_fn
        elif args.kb_type == "bu":
            def planner_fn(task: str, tools, managed_agents) -> dict:
                bu_plan = bu_dynamic_proposal_planning(
                    example     = example,
                    top_k       = args.top_k,
                    model_name  = model_name,
                    key         = key,
                    url         = url,
                    model       = model,
                    slm         = slm,
                    force_depth = args.bu_depth,
                )
                return {"plan": bu_plan or "", "examples": {}}

            agent.planner_fn = planner_fn
    final_result = agent.run(augmented_question, additional_knowledge=_additional_knowledge_for_planning)
    agent_memory = agent.write_memory_to_messages(summary_mode=True)
    final_result = prepare_response(
        augmented_question, agent_memory, reformulation_model=model
    )
    output = str(final_result)
    print("=" * 30 + "Final Output." + "=" * 30)
    print(f"output:{output}")
    print("=" * 30 + "Final Output." + "=" * 30)

    intermediate_steps = []
    for memory_step in agent.memory.steps:
        memory_step.model_input_messages = None
        step_dict = memory_step.dict()
        if isinstance(memory_step, ActionStep):
            step_dict["step_type"] = "action"
            step_dict.pop("model_output_message", None)
        elif isinstance(memory_step, TaskStep):
            step_dict["step_type"] = "task"
        elif isinstance(memory_step, PlanningStep):
            step_dict["step_type"] = "planning"
            step_dict.pop("model_output_message_facts", None)
            step_dict.pop("model_output_message_plan", None)
            logger.info(f"[PlanningStep] Facts:\n{memory_step.facts}")
            logger.info(f"[PlanningStep] Plan:\n{memory_step.plan}")
        else:
            step_dict["step_type"] = "unknown"
        intermediate_steps.append(step_dict)

    intermediate_steps_check = [str(step) for step in agent.memory.steps]
    parsing_error = (
        True
        if any(["AgentParsingError" in step for step in intermediate_steps_check])
        else False
    )

    iteration_limit_exceeded = (
        True
        if "Agent stopped due to iteration limit or time limit." in output
        else False
    )
    raised_exception = False

    # except Exception as e:
    #     logger.error(f"Error on task {example['task_id']}\n{e}")
    #     output = None
    #     intermediate_steps = []
    #     action_trajectory = []
    #     parsing_error = False
    #     iteration_limit_exceeded = False
    #     exception = e
    #     raised_exception = True
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotated_example = {
        "agent_name": model.model_id,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "true_answer": example["true_answer"],
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
        "task": example["task"],
        "task_id": example["task_id"],
    }
    append_answer(annotated_example, answers_file, jsonl_lock)


def get_examples_to_answer(
    answers_file, eval_df, selected_tasks=None, level="all", debug=False
) -> List[dict]:
    logger.info(f"Loading answers from {answers_file}...")
    try:
        done_questions = pd.read_json(answers_file, lines=True)["task_id"].tolist()
        logger.info(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        logger.info("Error when loading records: ", e)
        logger.info("No usable records! ▶️ Starting new.")
        done_questions = []

    if level == "all":
        filtered_df = eval_df
    else:
        filtered_df = eval_df[eval_df["task"] == level]

    if selected_tasks:
        if isinstance(selected_tasks[0], int):
            filtered_df = eval_df.iloc[selected_tasks]
        else:
            filtered_df = eval_df[eval_df["task_id"].isin(selected_tasks)]

    if debug:
        done_questions = []
    return [
        row.to_dict()
        for idx, row in filtered_df.iterrows()
        if row["task_id"] not in done_questions
    ]


def main():
    args = parse_args()
    logger.info(f"Starting run with arguments: {args}")

    answers_file = f"output/{SET}/{args.run_name}.jsonl"
    selected_tasks = process_selected_tasks_param(args.selected_tasks)
    level = args.level
    tasks_to_run = get_examples_to_answer(
        answers_file, eval_df, selected_tasks, level, args.debug
    )

    if args.slm:
        together_id, together_key, together_url, together_wrapper = get_together_model(None)
        model = together_wrapper(
            together_id,
            custom_role_conversions=custom_role_conversions,
            max_completion_tokens=8192,
            api_key=together_key,
            api_base=together_url,
            temperature=0.7,
        )
        model_search = together_wrapper(
            together_id,
            custom_role_conversions=custom_role_conversions,
            max_completion_tokens=8192,
            api_key=together_key,
            api_base=together_url,
            temperature=0.7,
        )
    else:
        model, model_search = None, None
    non_tool_probs = [
        "ec09fa32-d03f-4bf8-84b0-1f16922c3ae4",  # 1
        # "cffe0e32-c9a6-4c52-9877-78ceb4aaa9fb", #2
        "2d83110e-a098-4ebb-9987-066c06fa42d0",  # 3
        # "27d5d136-8563-469e-92bf-fd103c28b57c",  # 4
        # "dc28cf18-6431-458b-83ef-64b3ce566c10",  # 5
        "42576abe-0deb-4869-8c63-225c2d75a95a",  # 6
        # "6f37996b-2ac7-44b0-8e68-6d28256631b4",  # 7
        # "4b650a35-8529-4695-89ed-8dc7a500a498",  # 8
        # "c714ab3a-da30-4603-bacd-d008800188b9",  # 9
        # "3cef3a44-215e-4aed-8e3b-b1e3f08063b7",  # 10
        "e142056d-56ab-4352-b091-b56054bd1359",  # 11
        "50ad0280-0819-4bd9-b275-5de32d3b5bcb",  # 12
        "50ec8903-b81f-4257-9450-1085afd2c319",  # 13
    ]
    if args.debug or args.concurrency == 1:
        for example in tasks_to_run:
            if example["task_id"] not in non_tool_probs:
                continue
            answer_single_question(
                example,
                args,
                args.model_id,
                args.model_id_search,
                answers_file,
                args.debug,
                args.agent_kb,
                args.apply_student,
                args.apply_teacher,
                args.slm,
                model,
                model_search,
                args.use_sub_ex,
            )
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
            futures = [
                exe.submit(
                    answer_single_question,
                    example,
                    args,
                    args.model_id,
                    args.model_id_search,
                    answers_file,
                    args.debug,
                    args.agent_kb,
                    args.apply_student,
                    args.apply_teacher,
                    args.slm,
                    model,
                    model_search,
                    args.use_sub_ex,
                )
                for example in tasks_to_run
            ]
            for f in tqdm(
                as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"
            ):
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"Task failed: {str(e)}")

    logger.info("All tasks processed.")


if __name__ == "__main__":
    main()
