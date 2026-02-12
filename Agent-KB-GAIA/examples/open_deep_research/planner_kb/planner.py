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

from typing import Any, Dict, List, Literal, Optional

import logging

logger = logging.getLogger(__name__)

Mode = Literal["kci", "approach", "plan", "all"]


def build_examples(entities, planning_field="plan"):
    lines = []
    for entity in entities:
        task = entity.get("task") or entity.get("query") or entity.get("question")
        plan = entity.get(planning_field, "")
        lines.append(f"Similar task:{task}\nPlan: {plan}\n")
    return "\n".join(lines)


def planning_task(
    example,
    augmented_question,
    model_name,
    key,
    url,
    model,
    slm,
    retrieval_method,
    top_k,
    is_augmented=True,
):
    if retrieval_method is None:
        logger.info(
            f"planning_task - retrieval_method is None.(is_augmented={is_augmented})"
        )
        planning_prompt_template = load_prompts(
            path="/home/huijeong/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/planner_prompts.yaml"
        )
        planning_prompt_template = planning_prompt_template["planning_prompt"]
        planning_prompt = populate_template(
            planning_prompt_template,
            variables={"task": augmented_question},
        )
    else:
        logger.info(
            f"planning_task - retrieval_method is not None.(is_augmented={is_augmented})"
        )
        retrieval_results = retrieval_method(example["question"], top_k=top_k)
        # logger.info(f"Retrieved retrieval_results:\n {retrieval_results}")
        planning_prompt_template = load_prompts(
            path="/home/huijeong/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/planner_prompts.yaml"
        )
        planning_prompt_template = planning_prompt_template[
            "planning_with_examples_prompt"
        ]
        examples = build_examples(
            retrieval_results, "plan" if is_augmented else "agent_planning"
        )
        logger.info(f"Retrieved examples:\n {examples}")
        planning_prompt = populate_template(
            planning_prompt_template,
            variables={
                "task": augmented_question,
                "examples": examples,
            },
        )
    planning_str = call_model(
        query=planning_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info(f"planning_task - planning_str: {planning_str}")
    return planning_str


def build_similar_task_blocks(
    similars: List[Any],
    max_items: int = 5,
    mode: Mode = "kci",
) -> str:
    """
    유사태스크(TaskResponse 결과)들을 단계별(prompt별)로 필요한 필드만 포함해 텍스트 블록 구성.

    - mode="kci": Task + Knowledge + Constraints
    - mode="approach": Task + Approach
    - mode="plan": Task + Plan
    - mode="all": Task + Knowledge + Constraints + Approach + Plan
    """
    lines: List[str] = []

    for i, s in enumerate(similars[:max_items], start=1):
        d = _as_dict(s)

        task = d.get("task") or d.get("query") or d.get("question") or ""
        knowledge = d.get("knowledge") or ""
        constraints = d.get("contraints_instructions") or ""
        approach = d.get("approach") or ""
        plan = d.get("plan") or d.get("agent_planning") or ""

        parts: List[str] = [f"[Similar #{i}]"]

        # 항상 task는 포함
        if task:
            parts.append(f"Task: {task}".strip())

        if mode == "kci":
            if knowledge:
                parts.append(f"Knowledge:\n{knowledge}".strip())
            if constraints:
                parts.append(f"Constraints/Instructions:\n{constraints}".strip())

        elif mode == "approach":
            if approach:
                parts.append(f"Approach:\n{approach}".strip())

        elif mode == "plan":
            if plan:
                parts.append(f"Plan:\n{plan}".strip())

        elif mode == "all":
            if knowledge:
                parts.append(f"Knowledge:\n{knowledge}".strip())
            if constraints:
                parts.append(f"Constraints/Instructions:\n{constraints}".strip())
            if approach:
                parts.append(f"Approach:\n{approach}".strip())
            if plan:
                parts.append(f"Plan:\n{plan}".strip())

        lines.append("\n".join(parts).strip())

    return "\n\n".join(lines).strip()


def progressive_planning_task(
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

    retrieval_results = retrieval_method(example["question"], top_k=top_k)
    planning_prompt_template = load_prompts(
        path="/home/huijeong/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/planner_prompts.yaml"
    )

    # 1) knowledge + contraints_instructions
    progressive_kci_prompt_template = planning_prompt_template["progressive_kci_prompt"]
    similar_blocks_kci = build_similar_task_blocks(similars, mode="kci")
    progressive_kci_prompt = populate_template(
        progressive_kci_prompt_template,
        variables={
            "task": augmented_question,
            "similar_blocks": similar_blocks_kci,
        },
    )
    kci_str = call_model(
        query=progressive_kci_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"Generated KCI Prompt:\n{progressive_kci_prompt}")
    logger.info(f"Generated KCI:\n{kci_str}")
    logger.info("=" * 100)

    # 2) approach
    progressive_approach_prompt_template = planning_prompt_template[
        "progressive_approach_prompt"
    ]
    similar_blocks_approach = build_similar_task_blocks(similars, mode="approach")
    progressive_approach_prompt = populate_template(
        progressive_approach_prompt_template,
        variables={
            "task": augmented_question,
            "kci": kci_str,
            "similar_blocks": similar_blocks_approach,
        },
    )
    approach_str = call_model(
        query=progressive_approach_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"Generated Approach Prompt:\n{progressive_approach_prompt}")
    logger.info(f"Generated Approach:\n{approach_str}")
    logger.info("=" * 100)

    # 3) plan
    progressive_plan_prompt_template = planning_prompt_template[
        "progressive_plan_prompt"
    ]
    similar_blocks_plan = build_similar_task_blocks(similars, mode="plan")
    progressive_plan_prompt = populate_template(
        progressive_plan_prompt_template,
        variables={
            "task": augmented_question,
            "kci": kci_str,
            "approach": approach_str,
            "similar_blocks": similar_blocks_approach,
        },
    )
    plan_str = call_model(
        query=progressive_plan_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"Generated Plan Prompt:\n{progressive_plan_prompt}")
    logger.info(f"Generated Plan:\n{plan_str}")
    logger.info("=" * 100)

    return plan_str
