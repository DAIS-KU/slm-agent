from __future__ import annotations


from smolagents.agents import populate_template
import yaml
from agent_kb.agent_kb_utils import call_model
import ast
import json
import re
from typing import Any, Dict, List

from .inter_mece import *
from .intra_mece import *
from .mece_utils import load_prompts

import logging

logger = logging.getLogger(__name__)


def build_entities_example_string_no_actions(
    entity: Dict[str, Any],
    example_num: int = 1,
    *,
    indent: int = 4,
    ensure_ascii: bool = False,
) -> str:
    """
    TRANSFORM_SCHEMA 형태의 entity(dict)를 예시 포맷 문자열로 변환하되,
    출력에서는 Actions를 제외한다.

    Expected entity shape:
    {
      "task_id": str,
      "task": str,
      "subtasks": [
        {"subgoal": str, "rationale": str, "actions": [str, ...]},
        ...
      ]
    }

    Output:
      Example 1:
         - Task: ...
         - [{
              "Subgoal 1": "...",
              "Rationale 1": "..."
          }, ...]
    """
    task = str(entity.get("task", ""))
    subtasks: List[Dict[str, Any]] = entity.get("subtasks") or []
    if not isinstance(subtasks, list):
        raise TypeError("entity['subtasks'] must be a list")

    def _q(s: str) -> str:
        s = "" if s is None else str(s)
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        if ensure_ascii:
            s = s.encode("unicode_escape").decode("ascii")
        return f'"{s}"'

    ind1 = " " * indent
    ind2 = " " * (indent * 2)
    ind3 = " " * (indent * 3)

    lines: List[str] = []
    lines.append(f"Example {example_num}:")
    lines.append(f"{ind1}- Task: {task}")
    lines.append(f"{ind1}- [")

    for i, st in enumerate(subtasks, start=1):
        subgoal = st.get("subgoal", "")
        rationale = st.get("rationale", "")

        lines.append(f"{ind2}{{")
        lines.append(f"{ind3}{_q(f'Subgoal {i}')}: {_q(subgoal)},")
        lines.append(f"{ind3}{_q(f'Rationale {i}')}: {_q(rationale)}")

        if i < len(subtasks):
            lines.append(f"{ind2}}},")
        else:
            lines.append(f"{ind2}}}")

    lines.append(f"{ind1}]")
    return "\n".join(lines)


def build_many_entities_examples_no_actions(
    entities: List[Dict[str, Any]],
    *,
    start_example_num: int = 1,
    indent: int = 4,
    ensure_ascii: bool = False,
    separator: str = "\n\n",
) -> str:
    """여러 entity를 Example 1/2/3...로 연속 출력 (Actions 제외)."""
    parts: List[str] = []
    for idx, e in enumerate(entities, start=start_example_num):
        parts.append(
            build_entities_example_string_no_actions(
                e, example_num=idx, indent=indent, ensure_ascii=ensure_ascii
            )
        )
    return separator.join(parts)


def build_rationale_examples(
    entities, step_field="actions", rationale_field="rationale"
):
    step_and_rationaleses = []

    for entity in entities:
        lines = []
        task = entity.get("task") or entity.get("query") or entity.get("question")
        lines.append(f"Similar task:{task}")

        subtasks = entity.get("subtasks", [])
        for i, sub in enumerate(subtasks, start=1):
            subgoal = sub.get("subgoal", "")
            rationale = sub.get(rationale_field, "")

            lines.append(f"{i}. {subgoal}".rstrip())
            if rationale:
                lines.append(f"reason: {rationale}")

            actions = sub.get(step_field, [])
            for j, action in enumerate(actions, start=1):
                lines.append(f"  - {action}")

        step_and_rationaleses.append("\n".join(lines))

    return "\n".join(step_and_rationaleses)


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
):
    if retrieval_method is None:
        logger.info(f"decompose_task - retrieval_method is None")
        task_decomposition_prompt_template = load_prompts(
            path="/home/jovyan/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/rationale_planner_prompts.yaml"
        )
        task_decomposition_prompt_template = task_decomposition_prompt_template[
            "task_decomposition_and_planning_with_icl_examples_prompt"
        ]
        task_decomposition_prompt = populate_template(
            task_decomposition_prompt_template,
            variables={"task": augmented_question},
        )
    else:
        logger.info(f"decompose_task - retrieval_method is not None")
        rationale_retrieval_results = retrieval_method(example["question"], top_k=top_k)
        task_decomposition_prompt_template = load_prompts(
            path="/home/jovyan/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/rationale_planner_prompts.yaml"
        )
        step_rationale_examples = build_many_entities_examples_no_actions(
            rationale_retrieval_results
        )
        task_decomposition_prompt_template = task_decomposition_prompt_template[
            "task_decomposition_and_planning_with_retrieval_examples_prompt"
        ]
        task_decomposition_prompt = populate_template(
            task_decomposition_prompt_template,
            variables={
                "task": augmented_question,
                "retrieval_examples": step_rationale_examples,
            },
        )
    task_decomposition_str = call_model(
        query=task_decomposition_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info(f"decompose_task - task_decomposition_str: {task_decomposition_str}")
    return task_decomposition_str


def decompose_task_single(
    example,
    augmented_question,
    model_name,
    key,
    url,
    model,
    slm,
    retrieval_method,
    top_k,
    mode="max_surprise",
    outline=None,
    subtask_str_only=True,
):
    if retrieval_method is None:
        logger.info(f"decompose_task_single - retrieval_method is None")
        task_decomposition_prompt_template = load_prompts(
            path="/home/jovyan/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/rationale_planner_prompts.yaml"
        )
        if outline:
            task_decomposition_prompt_template = task_decomposition_prompt_template[
                "task_decomposition_and_planning_with_outline_and_icl_examples_prompt"
            ]
            task_decomposition_prompt = populate_template(
                task_decomposition_prompt_template,
                variables={"task": augmented_question, "approach": outline},
            )
        else:
            task_decomposition_prompt_template = task_decomposition_prompt_template[
                "task_decomposition_and_planning_with_icl_examples_prompt"
            ]
            task_decomposition_prompt = populate_template(
                task_decomposition_prompt_template,
                variables={"task": augmented_question},
            )
    else:
        logger.info(f"decompose_task_single - retrieval_method is not None")
        rationale_retrieval_results = retrieval_method(example["question"], top_k=top_k)
        task_decomposition_prompt_template = load_prompts(
            path="/home/jovyan/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/rationale_planner_prompts.yaml"
        )
        step_rationale_examples = build_many_entities_examples_no_actions(
            rationale_retrieval_results
        )
        if outline:
            task_decomposition_prompt_template = task_decomposition_prompt_template[
                "task_decomposition_and_planning_with_outline_and_retrieval_examples_prompt"
            ]
            task_decomposition_prompt = populate_template(
                task_decomposition_prompt_template,
                variables={
                    "task": augmented_question,
                    "approach": outline,
                    "retrieval_examples": step_rationale_examples,
                },
            )
        else:
            task_decomposition_prompt_template = task_decomposition_prompt_template[
                "task_decomposition_and_planning_with_retrieval_examples_prompt"
            ]
            task_decomposition_prompt = populate_template(
                task_decomposition_prompt_template,
                variables={
                    "task": augmented_question,
                    "retrieval_examples": step_rationale_examples,
                },
            )

    logger.info(f"decompose_task_single (mode {mode})")
    score = None
    if mode == "min_surprise" or mode == "max_surprise":
        engine = SurpriseInterMeceEngine(
            tm=model,
            call_model_fn=call_model,
            call_model_kwargs={
                "model_name": model_name,
                "key": key,
                "url": url,
                "model": model,
                "slm": slm,
            },
            max_length=2048,
        )
        cands = engine.pick_best(
            task_text=augmented_question,
            outline_text=outline,
            task_decomposition_prompt=task_decomposition_prompt,
            num_samples=8,
            objective="min" if mode == "min_surprise" else "max",
            return_topk=1,
        )
        best = cands[0]
        score = best.surprise.total_surprise
        logger.info(f"best.subtasks: {best.subtasks}")
        logger.info(f"best.surprise.total_surprise: {best.surprise.total_surprise}")
    elif mode == "sim":
        engine = SimInterMeceEngine(
            tm=model,
            call_model_fn=call_model,
            call_model_kwargs={
                "model_name": model_name,
                "key": key,
                "url": url,
                "model": model,
                "slm": slm,
            },
            max_length=2048,
        )
        cands = engine.pick_best(
            task_text=augmented_question,
            outline_text=outline,
            task_decomposition_prompt=task_decomposition_prompt,
            num_samples=8,
            return_topk=1,
        )
        best = cands[0]
        score = best.mece.inter_mece
        logger.info(f"best.subtasks: {best.subtasks}")
        logger.info(f"best.score: {best.subtasks}")
        logger.info(f"best.mece.redundancy: {best.mece.redundancy}")
    elif mode == "entropy":
        engine = EntropyInterMeceEngine(
            tm=model,
            call_model_fn=call_model,
            call_model_kwargs={
                "model_name": model_name,
                "key": key,
                "url": url,
                "model": model,
                "slm": slm,
            },
            max_length=2048,
        )
        cands = engine.pick_best(
            task_text=augmented_question,
            outline_text=outline,
            task_decomposition_prompt=task_decomposition_prompt,
            num_samples=8,
            return_topk=1,
        )
        best = cands[0]
        score = best.entropy.pairwise_js_mean
        logger.info(f"best.subtasks: {best.subtasks}")
        logger.info(f"best.entropy.pairwise_js_mean: {best.entropy.pairwise_js_mean}")
    else:
        raise NotImplementedError(f"decompose_task_single (mode {mode})")

    task_decomposition_str = call_model(
        query=task_decomposition_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info(
        f"decompose_task_single - task_decomposition_str: {task_decomposition_str}"
    )
    return task_decomposition_str if subtask_str_only else task_decomposition_str, score


def decompose_task_multiple(
    example,
    augmented_question,
    model_name,
    key,
    url,
    model,
    slm,
    retrieval_method,
    top_k,
    outline_mode="direction_only",
    inter_mode="max_surprise",
    multiple_n=12,
    multiple_k=5,
):
    engine = OutlineMeceEngine(
        tm=model,
        call_model_fn=call_model,
        call_model_kwargs=dict(
            model_name=model_name,
            key=key,
            url=url,
            model=model,
            slm=slm,
        ),
    )
    picked = engine.sample_and_pick(
        task_text=augmented_question, n=multiple_n, k=multiple_k, mode=outline_mode
    )
    outlines = picked.outlines
    logger.info(f"outline candidates: {outlines}")

    candidates = []
    for outline in outlines:
        decomposition_str, score = decompose_task_single(
            example=example,
            augmented_question=augmented_question,
            model_name=model_name,
            key=key,
            url=url,
            model=model,
            slm=slm,
            retrieval_method=retrieval_method,
            top_k=top_k,
            mode=inter_mode,
            outline=outline,
            subtask_str_only=False,
        )
        candidates.append((decomposition_str, score))
    candidates.sort(key=lambda c: c[1], reverse=True)
