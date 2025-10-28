import argparse
import json
import os
import threading
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, TypedDict, Union
from collections import Counter
import logging
import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login

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
from scripts.automodel import get_api_model, process_selected_tasks_param, prepare_model_kwargs

from agent_kb.agent_kb_utils import AKBClient, call_model

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
    "shutil"
]


parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
env_path = os.path.join(parent_dir, '.env')

load_dotenv(dotenv_path=env_path, override=True)
login(os.getenv("HF_TOKEN"))

logger = logging.getLogger(__name__)

jsonl_lock = threading.Lock()
trajectory_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--model-id-search", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--level", type=str, default="all", choices=["all", "1", "2", "3"],)
    parser.add_argument("--selected-tasks", default=None, nargs='*', help="Tasks to run: specify single or multiple indices (--selected-tasks 1 or --selected-tasks 1 2 5), a single task ID, or a path to a text file with one task ID per line")
    # infer params
    parser.add_argument('--planning_interval', type=int, default=1, help='Number of rollouts per state.')
    parser.add_argument("--max_steps", type=int, default=12, help="Maximum number of steps for ReAct agent.")
    parser.add_argument("--temperature", default=None, type=float, help= "The temperature for llm generation.")
    parser.add_argument('--top_p', default=None, type=float, help="The top_p for llm generation.")
    parser.add_argument('--search_reflection', action='store_true', help='Enable reflection')
    # agent_kb params
    parser.add_argument('--agent_kb', action='store_true', help='Enable knowledge base retrieval')
    parser.add_argument('--apply_student', action='store_true', help='Enable student correction')
    parser.add_argument('--apply_teacher', action='store_true', help='Enable teacher correction')
    parser.add_argument('--slm', action='store_true', help='Enable SLM agent')
    parser.add_argument('--retrieval_type', type=str, default="hybrid", help="search type")
    parser.add_argument('--top_k', type=int, default=3, help="top_k retrieval")
    parser.add_argument('--model_name_retrieval', type=str, default="gpt-4.1", help="agent kb model choice")
    
    return parser.parse_args()

logger.warning("Make sure you deactivated Tailscale VPN, else some URLs will be blocked!")

USE_OPEN_MODELS = False

SET = "validation"

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all", trust_remote_code=True)[SET]
eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})

def preprocess_file_paths(row):
    if len(row["file_name"]) > 0:
        row["file_name"] = f"data/gaia/{SET}/" + row["file_name"]
    return row

eval_ds = eval_ds.map(preprocess_file_paths)
eval_df = pd.DataFrame(eval_ds)

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
serp_api_key=os.getenv("SERP_API_KEY")
BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": serp_api_key,
    "num": 10
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)

def create_agent_hierarchy(model: Model, model_search: Model, args, debug=False):
    crawler = SimpleCrawler(serpapi_key=os.getenv("SERP_API_KEY"))
    text_limit = 100000

    search_types = ['wiki', 'google', 'baidu', 'bing', 'duckduckgo']
    search_tools = [SearchTool(search_type=st, reflection=False) for st in search_types]
    
    WEB_TOOLS = [
        CrawlerReadTool(crawler),
        CrawlerArchiveSearchTool(crawler),
        TextInspectorTool(model, text_limit),
    ]
    WEB_TOOLS += search_tools

    text_webbrowser_agent = ToolCallingAgent(
        model=model_search,
        tools=WEB_TOOLS,
        max_steps=args.max_steps,
        verbosity_level=2,
        planning_interval=args.planning_interval,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
        debug=debug,
        agent_kb=args.agent_kb,
        top_k=args.top_k,
        retrieval_type=args.retrieval_type,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""
    manager_agent = CodeAgent(
        model=model,
        tools=[VisualInspectorTool(model, text_limit), AudioInspectorTool(model, text_limit), TextInspectorTool(model, text_limit)],
        max_steps=args.max_steps,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=args.planning_interval,
        managed_agents=[text_webbrowser_agent],
        debug=debug,
        agent_kb=args.agent_kb,
        top_k=args.top_k,
        retrieval_type=args.retrieval_type,
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

def answer_single_question(example, args, model_id, model_id_search, answers_file, debug=False, retrieval=False, apply_student=False, apply_teacher=False, slm=False):

    if slm:
        key, url=None, None
        model = TransformersModel(
            model_id=model_id,
            device_map="auto",          
            # trust_remote_code=True, 
            torch_dtype="auto", 
            # max_new_tokens=2048, 
            temperature=0.1, 
            # token="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"
        )
        model_search = TransformersModel(
            model_id=model_id_search,
            device_map="auto",          
            # trust_remote_code=True, 
            torch_dtype="auto", 
            # max_new_tokens=2048, 
            temperature=0.1, 
            # token="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"
        )
    else:
        model_name, key, url, model_wrapper = get_api_model(model_id)
        model_name_search, key_search, url_search, model_wrapper_search = get_api_model(model_id_search)

        kwargs = prepare_model_kwargs(model_id, args)
        kwargs_search = prepare_model_kwargs(model_id_search, args)

        model = model_wrapper(
            model_name,
            custom_role_conversions=custom_role_conversions,
            max_completion_tokens=8192,
            api_key=key,
            api_base=url,
            **kwargs
        )

        model_search = model_wrapper_search(
            model_name_search,
            custom_role_conversions=custom_role_conversions,
            max_completion_tokens=8192,
            api_key=key_search,
            api_base=url_search,
            **kwargs_search
        )

    document_inspection_tool = TextInspectorTool(model, 100000)
    audio_inspection_tool = AudioInspectorTool(model, 100000)
    visual_inspection_tool = VisualInspectorTool(model, 100000)

    agent = create_agent_hierarchy(model, model_search, args, debug)

    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist). 
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer!
Here is the task:
""" + example["question"]

    if example["file_name"]:
        if ".zip" in example["file_name"]:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool, audio_inspection_tool,
            )
        else:
            prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:"
            prompt_use_files += get_single_file_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool, audio_inspection_tool,
            )
        augmented_question += prompt_use_files

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        if apply_student:
            print("="*30+"Student retrieval started."+"="*30)
            akb_client = AKBClient()
            model_name_retrieval = args.model_name_retrieval
            # import prompts for agent kb
            with open("./agent_kb/prompts.yaml", "r") as f:
                prompts = yaml.safe_load(f)
            semantic_match_template = prompts["semantic_match_prompt"]

            student_agent_reason_template = prompts["student_agent_reason"]
            teacher_agent_reason_template = prompts["teacher_agent_reason"]

            student_agent_refine_template = prompts["student_agent_refine"]
            teacher_agent_refine_template = prompts["teacher_agent_refine"]
            
            student_reason = populate_template(
                student_agent_reason_template,
                variables={"user_query": example["question"]}
            )
            student_summary = call_model(query=student_reason, model_name=model_name_retrieval, key=key, url=url, model=model, slm=slm)
            print("="*30+"Student Reasoned"+"="*30) 
            print(f"student_summary:{student_summary}")           
            print("="*30+"Student Reasoned"+"="*30)

            retrieval_method = {
                "hybrid": akb_client.hybrid_search,
                "text": akb_client.text_search,
                "semantic": akb_client.semantic_search
            }[args.retrieval_type]

            student_retrieval_results = retrieval_method(student_summary, top_k=args.top_k)
            student_retrieval = ""
            for result in student_retrieval_results:
                student_retrieval += "\nSimilar task:\n"
                student_retrieval += result['query']
                # student_retrieval += "\nPlannings:\n"
                # student_retrieval += result['plan']
                student_retrieval += "\nSuggestions:\n"
                student_retrieval += result['agent_experience']
                
            student_refine = populate_template(
                student_agent_refine_template,
                variables={"knowledge": student_retrieval}
            )
            
            student_suggestions = call_model(query=student_refine, model_name=model_name_retrieval, key=key, url=url, model=model, slm=slm)
            print("="*30+"Student Refiened."+"="*30)
            print(f"student_refine:{student_refine}")
            print(f"student_suggestions:{student_suggestions}")
            print("="*30+"Student Refiened."+"="*30)

            final_result = agent.run(augmented_question, additional_knowledge=student_suggestions)
            agent_memory = agent.write_memory_to_messages(summary_mode=True)
            final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)
            output = str(final_result)
            print("="*30+"Student Output."+"="*30)
            print(f"output:{output}")
            print("="*30+"Student Output."+"="*30)

            output_query = populate_template(
                semantic_match_template,
                variables={
                    "question": example["question"],
                    "prediction": output,
                    "true_answer": example["true_answer"]
                }
            )

            score_result = question_scorer(output, example["true_answer"])
            semantic_check = call_model(query=output_query, model_name=model_name_retrieval, key=key, url=url, model=model, slm=slm)
            print("="*30+f"score_result: {score_result}, semantic_check: {semantic_check}"+"="*30)

            if apply_teacher and (not score_result) and (semantic_check != "true"):
            # if (not score_result) and (semantic_check == "false"):
                print("="*30 + "Teacher Called"+"="*30)
                akb_client = AKBClient()
                intermediate_steps=[]
                for memory_step in agent.memory.steps:
                    memory_step.model_input_messages = None
                    step_dict = memory_step.dict()
                    if isinstance(memory_step, ActionStep):
                        step_dict['step_type'] = 'action'
                        step_dict.pop('model_output_message', None)
                    elif isinstance(memory_step, TaskStep):
                        step_dict['step_type'] = 'task'
                    elif isinstance(memory_step, PlanningStep):
                        log_plan = step_dict.get('plan', None)
                        step_dict['step_type'] = 'planning'
                        step_dict.pop('model_output_message_facts', None)
                        step_dict.pop('model_output_message_plan', None)
                    else:
                        step_dict['step_type'] = 'unknown'
                    intermediate_steps.append(step_dict)

                annotated_example = {
                    "question": example["question"],
                    "prediction": output,
                    "intermediate_steps": intermediate_steps,
                }

                teacher_reason = populate_template(
                    teacher_agent_reason_template,
                    variables={"agent_log": str(annotated_example)}
                )

                summary = call_model(query=teacher_reason, model_name=model_name_retrieval, key=key, url=url, model=model, slm=slm)
                print("="*30 + "Teacher Reasoned."+"="*30)
                print(f"teacher_reason:{teacher_reason}")    
                print(f"summary:{summary}")                
                print("="*30 + "Teacher Reasoned."+"="*30)

                teacher_retrieval_results = retrieval_method(example["question"] + log_plan + summary, top_k=args.top_k)

                teacher_retrieval = ""
                for result in teacher_retrieval_results:
                    teacher_retrieval += "\nSimilar task:\n"
                    teacher_retrieval += result['query']
                    # teacher_retrieval += "\nPlannings:\n"
                    # teacher_retrieval += result['plan']
                    teacher_retrieval += "\nSuggestions:\n"
                    teacher_retrieval += result['agent_experience']

                teacher_refine = populate_template(
                    teacher_agent_refine_template,
                    variables={
                        "knowledge": teacher_retrieval,
                        "log_summary": summary
                    }
                )

                teacher_suggestions = call_model(query=teacher_refine, model_name=model_name_retrieval, key=key, url=url, model=model, slm=slm)
                print("="*30 + "Teacher Refined."+"="*30)
                print(f"teacher_refine:{teacher_refine}")    
                print(f"teacher_suggestions:{teacher_suggestions}")                
                print("="*30 + "Teacher Refined."+"="*30)

                final_result = agent.run(augmented_question, additional_knowledge=teacher_suggestions)
                agent_memory = agent.write_memory_to_messages(summary_mode=True)
                final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)
                output = str(final_result)
                print("="*30+"Student Final Output."+"="*30)
                print(f"output:{output}")
                print("="*30+"Student Output."+"="*30)
        else:
            if retrieval:
                print("="*30+"Retrieval only without correction."+"="*30)
                akb_client = AKBClient()
                model_name_retrieval = args.model_name_retrieval
                retrieval_method = {
                    "hybrid": akb_client.hybrid_search,
                    "text": akb_client.text_search,
                    "semantic": akb_client.semantic_search
                }[args.retrieval_type]
                
                retrieval_results = retrieval_method(augmented_question, top_k=args.top_k)
                retrieval = "You must reference below contents.:\n\n"
                for result in retrieval_results:
                    retrieval += "\nSimilar task:\n"
                    retrieval += result['query']
                    retrieval += "\nSuggestions:\n"
                    retrieval += result['agent_experience']

                final_result = agent.run(augmented_question, additional_knowledge=retrieval)
                agent_memory = agent.write_memory_to_messages(summary_mode=True)
                final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)
                output = str(final_result)
                print("="*30+"Retrieval only Final Output."+"="*30)
                print(f"output:{output}")
                print("="*30+"Retrieval only Output."+"="*30)
            else:
                print("="*30+"Nothing Final retrieved."+"="*30)
                final_result = agent.run(augmented_question)
                agent_memory = agent.write_memory_to_messages(summary_mode=True)
                final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)
                output = str(final_result)
                print("="*30+"Retrieval Nothing Final Output."+"="*30)
                print(f"output:{output}")
                print("="*30+"Retrieval Nothing Output."+"="*30)


        intermediate_steps=[]
        for memory_step in agent.memory.steps:
            memory_step.model_input_messages = None
            step_dict = memory_step.dict()
            if isinstance(memory_step, ActionStep):
                step_dict['step_type'] = 'action'
                step_dict.pop('model_output_message', None)
            elif isinstance(memory_step, TaskStep):
                step_dict['step_type'] = 'task'
            elif isinstance(memory_step, PlanningStep):
                step_dict['step_type'] = 'planning'
                step_dict.pop('model_output_message_facts', None)
                step_dict.pop('model_output_message_plan', None)

            else:
                step_dict['step_type'] = 'unknown'
            intermediate_steps.append(step_dict)

        intermediate_steps_check = [str(step) for step in agent.memory.steps]
        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps_check]) else False

        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

    except Exception as e:
        logger.error(f"Error on task {example['task_id']}\n{e}")
        output = None
        intermediate_steps = []
        action_trajectory = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True
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
        "search_agent_actions": agent.managed_agents['search_agent'].task_records,
    }
    append_answer(annotated_example, answers_file, jsonl_lock)

def get_examples_to_answer(answers_file, eval_df, selected_tasks=None, level='all', debug=False) -> List[dict]:
    logger.info(f"Loading answers from {answers_file}...")
    try:
        done_questions = pd.read_json(answers_file, lines=True)["task_id"].tolist()
        logger.info(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        logger.info("Error when loading records: ", e)
        logger.info("No usable records! ▶️ Starting new.")
        done_questions = []

    if level == 'all':
        filtered_df = eval_df
    else:
        filtered_df = eval_df[eval_df['task'] == level]

    if selected_tasks:
        if isinstance(selected_tasks[0], int):
            filtered_df = eval_df.iloc[selected_tasks]
        else:
            filtered_df = eval_df[eval_df['task_id'].isin(selected_tasks)]

    if debug:
        done_questions = []
    return [row.to_dict() for idx, row in filtered_df.iterrows() if row["task_id"] not in done_questions]

def main():
    args = parse_args()
    logger.info(f"Starting run with arguments: {args}")

    answers_file = f"output/{SET}/{args.run_name}.jsonl"
    selected_tasks = process_selected_tasks_param(args.selected_tasks)
    level = args.level
    tasks_to_run = get_examples_to_answer(answers_file, eval_df, selected_tasks, level, args.debug)
    # tasks_ids_to_solve= [
    #     "e1fc63a2-da7a-432f-be78-7c4a95598703",
    #     "ec09fa32-d03f-4bf8-84b0-1f16922c3ae4",
    #     "cffe0e32-c9a6-4c52-9877-78ceb4aaa9fb",
    #     "2d83110e-a098-4ebb-9987-066c06fa42d0",
    #     "5cfb274c-0207-4aa7-9575-6ac0bd95d9b2",
    #     "27d5d136-8563-469e-92bf-fd103c28b57c",
    #     "dc28cf18-6431-458b-83ef-64b3ce566c10",
    #     "42576abe-0deb-4869-8c63-225c2d75a95a",
    #     "6f37996b-2ac7-44b0-8e68-6d28256631b4",
    #     "389793a7-ca17-4e82-81cb-2b3a2391b4b9",
    #     "4b650a35-8529-4695-89ed-8dc7a500a498",
    #     "c714ab3a-da30-4603-bacd-d008800188b9",
    #     "65afbc8a-89ca-4ad5-8d62-355bb401f61d",
    #     "3cef3a44-215e-4aed-8e3b-b1e3f08063b7"
    # ]

    if args.debug or args.concurrency == 1:
        for example in tasks_to_run:
            # if example['task_id'] not in tasks_ids_to_solve:
            #     continue
            answer_single_question(example, args, args.model_id, args.model_id_search, answers_file, args.debug, args.agent_kb, args.apply_student, args.apply_teacher, args.slm)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
            futures = [
                exe.submit(answer_single_question, example, args, args.model_id, args.model_id_search, answers_file, args.debug, args.agent_kb, args.apply_student, args.apply_teacher, args.slm)
                for example in tasks_to_run
            ]
            for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"Task failed: {str(e)}")

    logger.info("All tasks processed.")


if __name__ == "__main__":
    main()