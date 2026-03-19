from __future__ import annotations

from smolagents.agents import populate_template
from agent_kb.agent_kb_utils import call_model
from typing import Any, Dict, List, Optional

from .mece_utils import load_prompts

import logging

logger = logging.getLogger(__name__)


def build_similar_task_blocks(
    similars: List[Any], mode, max_items: int = 5, use_summary=False
) -> str:
    lines: List[str] = []

    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or d.get("query") or d.get("question") or ""
        plan_steps_only = d.get("plan_steps_only")

        parts: List[str] = []
        if mode == "plan_steps_only":
            parts.append(f"[Similar Task #{i}] {task}")
            parts.append(f"Plan: {plan_steps_only}\n")
        lines.append("\n".join(parts).strip())

    return "\n\n".join(lines).strip()


def build_similar_task_direction_blocks(similars: List[Any], max_items: int = 3) -> str:
    lines: List[str] = []

    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or d.get("query") or d.get("question") or ""
        problem_type = d.get("task_analysis").get("problem_type")
        domain = d.get("task_analysis").get("domain")
        knowledge = d.get("task_analysis").get("knowledge")
        approach = d.get("task_analysis").get("approach")
        lines.append(
            f"[Similar Task #{i}] {task}\nProblemType: {problem_type}\nDomain: {domain}\nKnowledge: {knowledge}\nApproach: {approach}\n"
        )
    return "\n\n".join(lines).strip()


def task_spec_approach_planning(
    example,
    augmented_question,
    model_name,
    key,
    url,
    model,
    slm,
    retrieval_method,
    top_k,
    observation=None,
):
    planning_prompt_template = load_prompts(
        path="/home/huijeong/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/planner_prompts.yaml"
    )

    # ====== [1] Generate knowledge, approach ====== #
    retrieval_results = retrieval_method(example["question"], top_k=top_k)
    examples = build_similar_task_direction_blocks(similars=retrieval_results)
    logger.info(f"Retrieved examples:\n {examples}")
    approach_prompt = populate_template(
        planning_prompt_template["approach_prompt"],
        variables={
            "task": augmented_question,
            "examples": examples,
        },
    )
    approach_str = call_model(
        query=approach_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"Generated Approach Prompt:\n{approach_prompt}")
    logger.info(f"Generated Approach:\n{approach_str}")
    logger.info("=" * 100)

    # ====== [2] Generate plan ====== #
    examples = build_similar_task_blocks(retrieval_results, mode="plan_steps_only")
    approach_to_plan_prompt = populate_template(
        planning_prompt_template["approach_to_plan_prompt"],
        variables={
            "task": augmented_question,
            "approach": approach_str,
            "observation": observation,
            "examples": examples,
        },
    )
    plan_str = call_model(
        query=approach_to_plan_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"Generated Plan Prompt:\n{approach_to_plan_prompt}")
    logger.info(f"Generated Plan:\n{plan_str}")
    logger.info("=" * 100)

    return plan_str
