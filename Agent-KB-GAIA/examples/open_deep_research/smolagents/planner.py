from smolagents.agents import populate_template
import yaml
from agent_kb.agent_kb_utils import call_model
import ast
from typing import List

# TODO
# 1. Use AKB Client instead of retrieval_method
# 2. Add action planning
# 3. Return only next plan, not whole plan


def parse_steps(output: str) -> list[str]:
    """
    "['Step 1: ...', 'Step 2: ...']" -> ['Step 1: ...', 'Step 2: ...']
    """
    text = output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    result = ast.literal_eval(text)
    if not isinstance(result, list) or not all(isinstance(x, str) for x in result):
        raise ValueError("Parsed result is not a list of strings.")
    return result


def load_prompts(path):
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)


def build_subtask_rationale_examples(entities, step_field, rationale_field):
    step_and_rationaleses = []
    for entity in entities:
        lines = []
        lines.append(f"Similar task:{entity['question']}")
        for i, (step, rationale) in enumerate(
            zip(entity[step_field], entity[rationale_field]), start=1
        ):
            lines.append(f"{i}. {plan}")
            lines.append(f"reason: {rationale}")
        step_and_rationales = "\n".join(lines)
        step_and_rationaleses.append(step_and_rationales)
    step_and_rationaleses_text = "\n".join(step_and_rationaleses)
    return step_and_rationaleses_text


def build_subtask_planning_rationale_examples(
    entities, subtask_number, step_field, rationale_field
):
    step_and_rationaleses = []
    for entit in entities:
        lines = []
        lines.append(f"Similar task:{entity['question']}")
        for i, (step, rationale) in enumerate(
            zip(
                entity[step_field][subtask_number],
                entity[rationale_field][subtask_number],
            ),
            start=1,
        ):
            lines.append(f"{i}. {plan}")
            lines.append(f"reason: {rationale}")
        step_and_rationales = "\n".join(lines)
        step_and_rationaleses.append(step_and_rationales)
    step_and_rationaleses_text = "\n".join(step_and_rationaleses)
    return step_and_rationaleses_text


def decompose_task(
    example,
    model_name,
    key,
    url,
    model,
    slm,
    retrieval_method,
    topk,
    return_as_str=False,
):
    if retrieval_method is None:
        task_decomposition_prompt_template = load_prompts(
            path="./rationale_planner_prompts.yaml"
        )["task_decomposition_prompt"]
        task_decomposition_prompt = populate_template(
            task_decomposition_prompt_template,
            variables={"task": example["question"]},
        )
    else:
        rationale_retrieval_results = retrieval_method(example["question"], top_k=top_k)
        task_decomposition_prompt_template = load_prompts(
            path="./rationale_planner_prompts.yaml"
        )["task_decomposition_with_examples_prompt"]
        step_rationale_examples = build_subtask_rationale_examples(
            rationale_retrieval_results, "Steps", "StepRationales"
        )
        task_decomposition_prompt = populate_template(
            task_decomposition_prompt_template,
            variables={
                "task": example["question"],
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
    if return_as_str:
        return task_decomposition
    else:
        extract_step_list_template = load_prompts(
            path="./rationale_planner_prompts.yaml"
        )["extract_step_list"]
        extract_step_list_prompt = populate_template(
            extract_step_lists_template, variables={"steps": task_decomposition_result}
        )
        extracted_steps = call_model(
            query=extract_step_list_prompt,
            model_name=model_name,
            key=key,
            url=url,
            model=model,
            slm=slm,
        )
        extracted_step_list = parse_steps(extracted_steps)
        return extracted_step_list


def subtask_planning(
    example,
    extracted_step_list,
    model_name,
    key,
    url,
    model,
    slm,
    retrieval_method,
    topk,
    return_as_str=False,
):
    subtask_plannings = []
    for curruent_sub_task_number, curruent_sub_task in enumerate(extracted_step_list):
        if retrieval_method is None:
            subtask_planning_prompt_template = load_prompts(
                path="./rationale_planner_prompts.yaml"
            )["subtask_planning_prompt"]
            subtask_planning_prompt = populate_template(
                task_decomposition_prompt_template,
                variables={
                    "task": example["question"],
                    "sub_tasks": extracted_step_list,
                    "curruent_sub_task": curruent_sub_task,
                },
            )
        else:
            subtask_planning_with_examples_prompt_template = load_prompts(
                path="./rationale_planner_prompts.yaml"
            )["subtask_planning_with_examples_prompt"]
            rationale_retrieval_results = retrieval_method(
                curruent_sub_task, top_k=top_k
            )
            step_rationale_examples = build_subtask_planning_rationale_examples(
                rationale_retrieval_results,
                curruent_sub_task_number,
                "Steps",
                "StepRationales",
            )
            task_decomposition = populate_template(
                task_decomposition_prompt_template,
                variables={
                    "task": example["question"],
                    "sub_tasks": extracted_step_list,
                    "curruent_sub_task": curruent_sub_task,
                    "planning_with_rationale_examples": step_rationale_examples,
                },
            )
        sub_task_planning_result = call_model(
            query=task_decomposition,
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
