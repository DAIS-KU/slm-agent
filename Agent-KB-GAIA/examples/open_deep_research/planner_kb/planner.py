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
    planning_field="plan",
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
        examples = build_examples(retrieval_results, planning_field)
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
    similars: List[Any], mode, max_items: int = 5, use_summary=False
) -> str:
    lines: List[str] = []

    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or d.get("query") or d.get("question") or ""
        domain = d.get("domain")
        skills = d.get("skills")
        objective = d.get("objective")
        knowledge = d.get("knowledge_summary") if use_summary else d.get("knowledge")
        constraints = (
            d.get("constratints_summary") if use_summary else d.get("constraints")
        )
        instructions = (
            d.get("instructions_summary") if use_summary else d.get("instructions")
        )
        approach = d.get("approach_summary") if use_summary else d.get("approach")
        plan = d.get("plan_summary") if use_summary else d.get("plan")

        parts: List[str] = [f"[Similar Task #{i}] {task}\n"]
        if mode == "knowledge":
            parts.append(f"Knowledge: {knowledge}")
        elif mode == "approach":
            parts.append(f"Approach:{approach}")
        elif mode == "plan":
            parts.append(f"Plan: {plan}")
        elif mode == "spec":
            parts.append(f"Domain:{domain}\nSkills:{skills}\nObjective:{objective}")

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
    use_summary=False,
):

    retrieval_results = retrieval_method(example["question"], top_k=top_k)
    planning_prompt_template = load_prompts(
        path="/home/huijeong/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/planner_prompts.yaml"
    )

    # 1) knowledge
    progressive_knowledge_prompt_template = planning_prompt_template[
        "progressive_knowledge_prompt"
    ]
    similar_blocks_knowledge = build_similar_task_blocks(
        retrieval_results, mode="knowledge", use_summary=use_summary
    )
    progressive_knowledge_prompt = populate_template(
        progressive_knowledge_prompt_template,
        variables={
            "task": augmented_question,
            "similar_blocks": similar_blocks_knowledge,
        },
    )
    raw_knowledge_str = call_model(
        query=progressive_kci_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    knowledge_str = call_model(
        query=f"Summerize the following text.\n\n{raw_knowledge_str}",
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"Generated Knowledge Prompt:\n{progressive_knowledge_prompt}")
    logger.info(f"Generated Knowledge:\n{knowledge_str}")
    logger.info("=" * 100)

    # 2) constraints, instructions
    progressive_ci_prompt_template = planning_prompt_template["progressive_ci_prompt"]
    progressive_ci_prompt = populate_template(
        progressive_ci_prompt_template,
        variables={
            "task": augmented_question,
        },
    )
    raw_ci_str = call_model(
        query=progressive_ci_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    ci_str = call_model(
        query=f"Summerize the following text.\n\n{raw_ci_str}",
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(
        f"Generated Constraitns/Instructions Prompt:\n{progressive_knowledge_prompt}"
    )
    logger.info(f"Generated Constraitns/Instructions:\n{ci_str}")
    logger.info("=" * 100)

    # 3) approach
    progressive_approach_prompt_template = planning_prompt_template[
        "progressive_approach_prompt"
    ]
    similar_blocks_approach = build_similar_task_blocks(
        retrieval_results, mode="approach", use_summary=use_summary
    )
    progressive_approach_prompt = populate_template(
        progressive_approach_prompt_template,
        variables={
            "task": augmented_question,
            "knowledge": knowledge_str,
            "constraints_instructions": ci_str,
            "similar_blocks": similar_blocks_approach,
        },
    )
    raw_approach_str = call_model(
        query=progressive_approach_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    approach_str = call_model(
        query=f"Summerize the following text.\n\n{raw_approach_str}",
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
    similar_blocks_plan = build_similar_task_blocks(
        retrieval_results, mode="plan", use_summary=use_summary
    )
    progressive_plan_prompt = populate_template(
        progressive_plan_prompt_template,
        variables={
            "task": augmented_question,
            "knowledge": knowledge_str,
            "constraints_instructions": ci_str,
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

    return plan_str, ci_str


def recontextulaized_planning_task(
    example: Dict[str, Any],
    augmented_question: str,
    model_name: str,
    key: str,
    url: str,
    model: Any,
    slm: Any,
    retrieval_method=None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    (1) goal task -> TaskSpec + (constraints,instructions) 추출 + 정리
    (2) similar tasks 각각 -> ref TaskSpec -> transfer(knowledge/approach) -> 통합 요약
    (3) goal + CI + transferred summary로 최종 plan 생성

    Returns a dict with plan + intermediate artifacts for debugging/inspection.
    """

    prompts = load_prompts(
        path="/home/huijeong/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/planner_prompts.yaml"
    )

    # ---- (1) Goal: TaskSpec 추출 ----
    extract_goal_prompt_template = prompts["extract_task_spec_prompt"]
    extract_goal_prompt = populate_template(
        extract_goal_prompt_template,
        variables={"task": augmented_question},
    )
    goal_task_spec = call_model(
        query=extract_goal_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"Generated TaskSpec:\n{goal_task_spec}")
    logger.info("=" * 100)

    # ---- (1) Goal: constraints/instructions 추출 ----
    ci_prompt_template = planning_prompt_template["progressive_ci_prompt"]
    ci_prompt = populate_template(
        ci_prompt_template,
        variables={
            "task": augmented_question,
        },
    )
    raw_ci_str = call_model(
        query=ci_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    ci_str = call_model(
        query=f"Summerize the following text.\n\n{raw_ci_str}",
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"Generated Constraitns/Instructions:\n{ci_str}")
    logger.info("=" * 100)

    # ---- (2) Transfer tasks ----
    retrieval_results: List[Dict[str, Any]] = []
    if retrieval_method is not None:
        retrieval_results = (
            retrieval_method(example.get("question", augmented_question), top_k=top_k)
            or []
        )
        ref_specs = build_similar_task_blocks(retrieval_results, mode="spec")
    else:
        logger.error(
            "recontextulaized_planning_task - retrieval_method is None, skipping similar tasks."
        )
        ref_specs = []

    transfers: List[Dict[str, Any]] = []
    for idx, ref_spec in enumerate(ref_specs):
        transter_ref_prompt_template = planning_prompt_template["transfer_ka_prompt"]
        transter_ref_prompt = populate_template(
            transter_ref_prompt_template,
            variables={"goal_task_spec": goal_task_spec, "ref_task_spec": ref_spec},
        )
        transfer = call_model(
            query=transter_ref_prompt,
            model_name=model_name,
            key=key,
            url=url,
            model=model,
            slm=slm,
        )
        logger.info("=" * 100)
        logger.info(f"Generated Tranfer Item:\n{transfer}")
        logger.info("=" * 100)
        transfers.append(transfer)
    proposed_ka = "\n".join(transfers)

    # ---- (3) 최종 knowledge and approach 생성 ----
    final_ka_prompt_template = planning_prompt_template["transfer_ka_prompt"]
    final_ka_prompt = populate_template(
        final_ka_prompt_template,
        variables={"task": augmented_question, "proposed_ka": proposed_ka},
    )
    final_ka = call_model(
        query=final_ka_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"Generated Final Knowledge and Approach:\n{final_ka}")
    logger.info("=" * 100)

    # ---- (3) 최종 Plan 생성 ----
    plan_prompt_template = planning_prompt_template["recontextualized_plan_prompt"]
    plan_prompt = populate_template(
        plan_prompt_template,
        variables={
            "task": augmented_question,
            "knowledge_approach": final_ka,
            "constraints_instructions": ci_str,
        },
    )
    plan_str = call_model(
        query=final_ka_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"Generated Final Paln:\n{plan_str}")
    logger.info("=" * 100)

    return plan_str, ci_str
