from smolagents.agents import populate_template
import yaml
from agent_kb.agent_kb_utils import call_model
import ast
import json
import re
from typing import List


def extract_steps(step_str, model_name, key, url, model, slm):
    print("extract steps.")
    extract_steps_prompt_template = load_prompts(
        path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/planner_kb/rationale_planner_prompts.yaml"
    )
    extract_step_number_prompt = populate_template(
        extract_steps_prompt_template["count_steps"],
        variables={"steps": step_str},
    )
    step_number = call_model(
        extract_step_number_prompt, model_name, key, url, model, slm
    )
    step_number = int(step_number)
    print(f"extract {step_number} steps.")

    steps = []
    for i in range(step_number):
        extract_specific_step_prompt = populate_template(
            extract_steps_prompt_template["extract_specific_step"],
            variables={"step_number": step_number + 1, "steps": step_str},
        )
        step = call_model(
            extract_specific_step_prompt, model_name, key, url, model, slm
        )
        steps.appen(step)
    return steps


def parse_steps(output: str) -> List[str]:
    print(f"parse_steps output:{output}")
    """
    Supported patterns (앞뒤에 잡소리 텍스트가 있어도 허용):

    1) JSON 배열:
       ["Step 1: ...", "Step 2: ..."]

    2) 파이썬 리스트/튜플 리터럴:
       ['Step 1: ...', 'Step 2: ...']
       ("Step 1: ...", "Step 2: ...")

    3) 그냥 문자열 나열:
       "Step 1: ...", "Step 2: ...", "Step 3: ..."

    4) Markdown 번호 리스트:
       1. Step 1 ...
          - sub bullet ...

       2. Step 2 ...
          - ...

    5) 'Step 1: ...', 'Step 2: ...' 처럼
       각 Step 헤더와 그 아래 여러 줄이 한 블록인 형식
    """
    text = output.strip()

    # 1) 코드블럭 제거 (``` ... ```)
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 2) 우선 [] / () 로 전체가 감싸져 있는 경우 처리
    stripped = text.lstrip()
    candidate = None
    result = None

    # (a) 텍스트 맨 앞이 [ 또는 ( 인 경우
    if stripped.startswith(("[", "(")):
        candidate = stripped
    else:
        # (b) "output str:(...)" 같은 형태: 콜론/등호 뒤에서부터 [ 또는 ( 찾기
        m = re.search(r"[:=]\s*([\[\(].*)$", text, re.DOTALL)
        if m:
            candidate = m.group(1).strip()

    if candidate:
        try:
            if candidate.startswith("["):
                # 먼저 JSON 시도
                try:
                    result = json.loads(candidate)
                except json.JSONDecodeError:
                    # 안 되면 파이썬 literal 로
                    result = ast.literal_eval(candidate)
            else:
                # "(" 로 시작하면 파이썬 tuple/list literal 로 간주
                result = ast.literal_eval(candidate)

            if isinstance(result, tuple):
                result = list(result)

            if isinstance(result, list) and all(isinstance(x, str) for x in result):
                return result
        except Exception:
            # candidate 파싱 실패하면 fallback 으로
            result = None

    # 3) Fallback 1: Markdown 번호 리스트 (1. ..., 2. ..., ...)
    #    각 번호 블록을 하나의 step 으로 취급
    numbered_pattern = r"^\s*\d+\.\s+(.+?)(?=^\s*\d+\.|\Z)"
    blocks = re.findall(
        numbered_pattern,
        text.strip(),
        flags=re.MULTILINE | re.DOTALL,
    )
    if blocks:
        steps = [b.strip() for b in blocks]
        return steps

    # 3.5) Fallback 1.5: "Step N:" 으로 시작하는 블록들
    #      예: Step 1: ... \n - ... \n\n Step 2: ...
    step_pattern = r"^Step\s+\d+:\s*(.*?)(?=^Step\s+\d+:\s*|\Z)"
    step_blocks = re.findall(
        step_pattern,
        text.strip(),
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if step_blocks:
        steps = [b.strip() for b in step_blocks]
        return steps

    # 4) Fallback 2: 따옴표로 된 문자열들을 전부 추출해서 리스트로 사용
    #   -> "Step 1: ...", "Step 2: ...", ... 같은 케이스 처리
    pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"|\'([^\'\\]*(?:\\.[^\'\\]*)*)\''
    matches = re.findall(pattern, text)

    if not matches:
        raise ValueError(f"Could not parse steps from: {output!r}")

    steps: List[str] = []
    for g1, g2 in matches:
        s = g1 or g2  # "..." 에서 잡힌 그룹 또는 '...' 에서 잡힌 그룹
        steps.append(s)

    if not steps:
        raise ValueError("No steps extracted from output.")

    return steps


def load_prompts(path):
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)
    return prompts


def build_rationale_examples(entities, step_field, rationale_field):
    step_and_rationaleses = []
    for entity in entities:
        lines = []
        task = entity.get("query") or entity.get("question")
        lines.append(f"Similar task:{task}")
        for i, (step, rationale) in enumerate(
            zip(entity[step_field], entity[rationale_field]), start=1
        ):
            lines.append(f"{i}. {step}")
            lines.append(f"reason: {rationale}")
        step_and_rationales = "\n".join(lines)
        step_and_rationaleses.append(step_and_rationales)
    step_and_rationaleses_text = "\n".join(step_and_rationaleses)
    return step_and_rationaleses_text


def decompose_task(
    example,
    augmented_question,
    model_name,
    key,
    url,
    model,
    slm,
    retrieval_method,
    top_k,
    return_as_str=False,
):
    if retrieval_method is None:
        print(f"decompose_task - retrieval_method is None")
        task_decomposition_prompt_template = load_prompts(
            path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/planner_kb/rationale_planner_prompts.yaml"
        )["task_decomposition_prompt"]
        task_decomposition_prompt = populate_template(
            task_decomposition_prompt_template,
            variables={"task": augmented_question},
        )
    else:
        print(f"decompose_task - retrieval_method is not None")
        rationale_retrieval_results = retrieval_method(example["question"], top_k=top_k)
        task_decomposition_prompt_template = load_prompts(
            path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/planner_kb/rationale_planner_prompts.yaml"
        )["task_decomposition_with_examples_prompt"]
        step_rationale_examples = build_rationale_examples(
            rationale_retrieval_results, "steps", "step_rationales"
        )
        task_decomposition_prompt = populate_template(
            task_decomposition_prompt_template,
            variables={
                "task": augmented_question,
                "decomposed_with_rationale_examples": step_rationale_examples,
            },
        )
    task_decomposition_result = call_model(
        query=task_decomposition_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    print(f"decompose_task - task_decomposition_result: {task_decomposition_result}")
    if return_as_str:
        return task_decomposition_result
    else:
        # extract_step_list_template = load_prompts(
        #     path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/planner_kb/rationale_planner_prompts.yaml"
        # )["extract_step_list"]
        # extract_step_list_prompt = populate_template(
        #     extract_step_list_template, variables={"steps": task_decomposition_result}
        # )
        # # print(f"extract_step_list_prompt: {extract_step_list_prompt}")
        # extracted_steps = call_model(
        #                 query=extract_step_list_prompt,
        #                 model_name=model_name,
        #                 key=key,
        #                 url=url,
        #                 model=model,
        #                 slm=slm,
        #             )
        # # print(f"extracted_steps: {extracted_steps}")
        extracted_step_list = parse_steps(task_decomposition_result)
        # extracted_step_list =  extract_steps(task_decomposition_result, model_name, key,url,model,slm)
        return extracted_step_list


def subtask_planning(
    example,
    augmented_question,
    extracted_step_list,
    model_name,
    key,
    url,
    model,
    slm,
    retrieval_method,
    top_k,
    return_as_str=False,
):
    subtask_plannings = []
    for curruent_sub_task_number, curruent_sub_task in enumerate(extracted_step_list):
        print(f"subtask planning #{curruent_sub_task_number}")
        if retrieval_method is None:
            subtask_planning_prompt_template = load_prompts(
                path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/planner_kb/rationale_planner_prompts.yaml"
            )["subtask_planning_prompt"]
            subtask_planning_prompt = populate_template(
                subtask_planning_prompt_template,
                variables={
                    "task": augmented_question,
                    "sub_tasks": extracted_step_list,
                    "curruent_sub_task": curruent_sub_task,
                },
            )
        else:
            subtask_planning_with_examples_prompt_template = load_prompts(
                path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/planner_kb/rationale_planner_prompts.yaml"
            )["subtask_planning_with_examples_prompt"]
            rationale_retrieval_results = retrieval_method(
                curruent_sub_task, top_k=top_k
            )
            step_rationale_examples = build_rationale_examples(
                rationale_retrieval_results,
                "steps",
                "step_rationales",
            )
            subtask_planning_prompt = populate_template(
                subtask_planning_with_examples_prompt_template,
                variables={
                    "task": augmented_question,
                    "sub_tasks": extracted_step_list,
                    "curruent_sub_task": curruent_sub_task,
                    "planning_with_rationale_examples": step_rationale_examples,
                },
            )
        sub_task_planning_result = call_model(
            query=subtask_planning_prompt,
            model_name=model_name,
            key=key,
            url=url,
            model=model,
            slm=slm,
        )
        subtask_plannings.append(sub_task_planning_result)
    if return_as_str:
        subtask_plannings_str = "\n".join(subtask_plannings)
        return subtask_plannings_str
    else:
        return subtask_plannings
