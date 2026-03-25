from __future__ import annotations

import json

from smolagents.agents import populate_template
from agent_kb.agent_kb_utils import call_model
from typing import Any, Dict, List, Literal, Optional

from .mece_utils import load_prompts

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Predefined constants for type_and_domain retrieval mode.
# These map to task_analysis.task_type and task_analysis.domain in the KB.
# Modify as needed.
# ---------------------------------------------------------------------------
TASK_TYPE_CONSTANTS: List[str] = [
'algebra', 'algorithm_design', 'analysis', 'arithmetic', 'calculus', 'causal_inference', 'chemical_analysis', 'circuit_analysis', 'classification', 'clinical_evaluation', 'code_analysis', 'combinatorial_analysis', 'computation_geometry', 'data_transformation', 'decision_making', 'decomposition', 'definition', 'document_analysis', 'estimation', 'formalization', 'game_analysis', 'graph_theory', 'group_theory', 'inequality_proof', 'legal_analysis', 'mapping', 'mathematical_reasoning', 'model_theory', 'networking', 'quantum_computation', 'statistical_analysis'
]

DOMAIN_CONSTANTS: List[str] = [
'acoustics', 'algebra', 'algorithm_design', 'analysis', 'biology', 'business', 'chemistry', 'computer_science', 'cryptography', 'education', 'engineering', 'ethics', 'experiment_design', 'games', 'healthcare', 'law', 'mathematics', 'medicine', 'music', 'networking', 'physics', 'political_science', 'protein_engineering', 'puzzles', 'representation_theory', 'robotics', 'sports', 'statistical_mechanics'
]


def build_action_augmented_plan_examples(
    similars: List[Any], max_items: int = 3
) -> str:
    """Build example blocks from the action_augmented_plan field (step + step_action)."""
    lines: List[str] = []
    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or d.get("query") or d.get("question") or ""
        items = d.get("action_augmented_plan") or []
        if not items:
            continue
        steps = []
        for j, item in enumerate(items, start=1):
            step = item.get("step", "")
            action = item.get("step_action", "")
            steps.append(f"  Step {j}: {step}\n  Action: {action}")
        if steps:
            lines.append(f"[Similar Task #{i}] {task}\n" + "\n".join(steps))
    return "\n\n".join(lines).strip()


def build_plan_and_subtask_examples(
    similars: List[Any], max_items: int = 3
) -> str:
    """Build example blocks from plan_only_steps (step) and subtasks_only_subtask (subtask)."""
    lines: List[str] = []
    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or d.get("query") or d.get("question") or ""
        steps = d.get("plan_only_steps") or []
        subtasks = d.get("subtasks_only_subtask") or []
        if not steps and not subtasks:
            continue
        block = [f"[Similar Task #{i}] {task}"]
        if steps:
            step_strs = [s.get("step", s) if isinstance(s, dict) else s for s in steps]
            block.append("Steps:\n" + "\n".join(f"  - {s}" for s in step_strs))
        if subtasks:
            sub_strs = [s.get("subtask", s) if isinstance(s, dict) else s for s in subtasks]
            block.append("Subtasks:\n" + "\n".join(f"  - {s}" for s in sub_strs))
        lines.append("\n".join(block))
    return "\n\n".join(lines).strip()


def build_action_augmented_subtask_examples(
    similars: List[Any], max_items: int = 3
) -> str:
    """Build example blocks from action_augmented_subtask (subtask + subtask_action)."""
    lines: List[str] = []
    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or d.get("query") or d.get("question") or ""
        items = d.get("action_augmented_subtask") or []
        if not items:
            continue
        steps = []
        for j, item in enumerate(items, start=1):
            subtask = item.get("subtask", "")
            action = item.get("subtask_action", "")
            steps.append(f"  Subtask {j}: {subtask}\n  Action: {action}")
        if steps:
            lines.append(f"[Similar Task #{i}] {task}\n" + "\n".join(steps))
    return "\n\n".join(lines).strip()


def build_plan_subtask_action_examples(
    similars: List[Any], max_items: int = 3
) -> str:
    """Build nested examples from the plan_subtask_action field.
    Each entry has: step, step_action, subtasks: [{subtask, subtask_action}].
    """
    lines: List[str] = []
    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or d.get("query") or d.get("question") or ""
        items = d.get("plan_subtask_action") or []
        if not items:
            continue
        block = [f"[Similar Task #{i}] {task}"]
        for j, step_item in enumerate(items, start=1):
            step = step_item.get("step", "")
            step_action = step_item.get("step_action", "")
            block.append(f"  ## Step {j}: {step}")
            if step_action:
                block.append(f"     Step Action: {step_action}")
            for k, st in enumerate(step_item.get("subtasks", []), start=1):
                subtask = st.get("subtask", "")
                subtask_action = st.get("subtask_action", "")
                block.append(f"     - Subtask {j}.{k}: {subtask}")
                if subtask_action:
                    block.append(f"       Action: {subtask_action}")
        lines.append("\n".join(block))
    return "\n\n".join(lines).strip()


def build_similar_task_blocks(
    similars: List[Any], mode, max_items: int = 5, use_summary=False
) -> str:
    lines: List[str] = []

    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or d.get("query") or d.get("question") or ""
        plan_only_steps = d.get("plan_only_steps") or []

        parts: List[str] = []
        if mode == "plan_steps_only":
            parts.append(f"[Similar Task #{i}] {task}")
            step_strs = [s.get("step", s) if isinstance(s, dict) else s for s in plan_only_steps]
            parts.append("Plan:\n" + "\n".join(f"  - {s}" for s in step_strs) + "\n")
        lines.append("\n".join(parts).strip())

    return "\n\n".join(lines).strip()


def build_similar_task_direction_blocks(similars: List[Any], max_items: int = 3) -> str:
    lines: List[str] = []

    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or d.get("query") or d.get("question") or ""
        task_type = d.get("task_analysis").get("task_type")
        domain = d.get("task_analysis").get("domain")
        knowledge = d.get("task_analysis").get("knowledge")
        approach = d.get("task_analysis").get("approach")
        lines.append(
            f"[Similar Task #{i}] {task}\nTaskType: {task_type}\nDomain: {domain}\nKnowledge: {knowledge}\nApproach: {approach}\n"
        )
    return "\n\n".join(lines).strip()


def classify_task_type_and_domain(
    task: str,
    planning_prompt_template: Dict[str, str],
    model_name: str,
    key: str,
    url: str,
    model: Any,
    slm: bool,
) -> Dict[str, List[str]]:
    """Call the LLM to classify task into predefined task_type and domain constants."""
    classification_prompt = populate_template(
        planning_prompt_template["task_classification_prompt"],
        variables={
            "task": task,
            "task_types": "\n".join(f"- {t}" for t in TASK_TYPE_CONSTANTS),
            "domains": "\n".join(f"- {d}" for d in DOMAIN_CONSTANTS),
        },
    )
    raw = call_model(
        query=classification_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    try:
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        task_types = result.get("task_type_normalized") or []
        domains = result.get("domain_normalized") or []
    except Exception as e:
        logger.warning(f"Task classification parsing failed: {e}. Raw: {raw}")
        task_types, domains = [], []
    logger.info(f"Task classification → task_type_normalized={task_types}, domain_normalized={domains}")
    return {"task_type_normalized": task_types, "domain_normalized": domains}


def proposal_planning(
    example,
    augmented_question,
    model_name,
    key,
    url,
    model,
    slm,
    retrieval_method,
    top_k,
    retrieval_option: Literal["task_text", "type_and_domain"] = "task_text",
    type_domain_retrieval_method=None,
):
    planning_prompt_template = load_prompts(
        path="/home/huijeong/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/planner_prompts.yaml"
    )

    # ====== [1] Retrieve similar tasks ====== #
    if retrieval_option == "type_and_domain" and type_domain_retrieval_method is not None:
        # Step 1: Classify current task into predefined constants
        classification = classify_task_type_and_domain(
            task=augmented_question,
            planning_prompt_template=planning_prompt_template,
            model_name=model_name,
            key=key,
            url=url,
            model=model,
            slm=slm,
        )
        task_types = classification["task_type_normalized"]
        domains = classification["domain_normalized"]

        # Step 2: Filter by type+domain overlap in the KB, rank by (overlap DESC, text_score DESC)
        retrieval_results = type_domain_retrieval_method(
            example["question"], task_types=task_types, domains=domains, top_k=top_k
        )
        if not retrieval_results:
            logger.warning(
                "type_and_domain retrieval returned no results; falling back to task_text retrieval."
            )
            retrieval_results = retrieval_method(example["question"], top_k=top_k)
    else:
        # Default: retrieve by task text similarity only
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

    examples = {
        "action_augmented_plan": build_action_augmented_plan_examples(retrieval_results),
        "plan_and_subtask": build_plan_and_subtask_examples(retrieval_results),
        "action_augmented_subtask": build_action_augmented_subtask_examples(retrieval_results),
        "plan_subtask_action": build_plan_subtask_action_examples(retrieval_results),
    }
    return {
        "plan": plan_str,
        "approach": approach_str,
        "retrieval_results": retrieval_results,
        "examples": examples,
    }
