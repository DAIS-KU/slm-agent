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


def build_examples(
    entities, planning_field="plan"
):
    lines = []
    for entity in entities:
        task = entity.get("task") or entity.get("query") or entity.get("question")
        plan = sub.get(planning_field, "")
        lines.append(f"Similar task:{task}\nPlan: {rationale}\n")
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
    is_augmented=True
):
    if retrieval_method is None:
        logger.info(f"planning_task - retrieval_method is None.(is_augmented={is_augmented})")
        planning_prompt_template = load_prompts(
            path="/home/huijeong/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/planner_prompts.yaml"
        )
        planning_prompt_template = planning_prompt_template[
            "planning_prompt"
        ]
        planning_prompt = populate_template(
            planning_prompt_template,
            variables={"task": augmented_question},
        )
    else:
        logger.info(f"planning_task - retrieval_method is not None.(is_augmented={is_augmented})")
        retrieval_results = retrieval_method(example["question"], top_k=top_k)
        planning_prompt_template = load_prompts(
            path="/home/huijeong/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/planner_prompts.yaml"
        )
        examples = build_examples(
            retrieval_results, "plan" if is_augmented else "agent_planning"
        )
        logger.info(f"Retrieved examples:\n {examples}")
        planning_prompt_template = planning_prompt_template[
            "planning_with_examples_prompt"
        ]
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