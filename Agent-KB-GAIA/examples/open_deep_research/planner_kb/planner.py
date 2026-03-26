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
TASK_TYPE_CONSTANTS: List[str] = ['analysis', 'computation', 'constraint_satisfaction', 'mathematics', 'reasoning']

DOMAIN_CONSTANTS: List[str] = ['artificial_intelligence', 'biology', 'chemistry', 'computer_science', 'engineering', 'linguistics', 'mathematics', 'physics', 'social_science', 'statistics']


def build_plan_subtask_examples(
    similars: List[Any], max_items: int = 3
) -> str:
    """Build examples showing step → subtask structure from plan_subtask_action (no actions)."""
    lines: List[str] = []
    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or ""
        items = d.get("plan_subtask_action") or []
        if not items:
            continue
        block = [f"[Similar Task #{i}] {task}"]
        for j, step_item in enumerate(items, start=1):
            step = step_item.get("step", "")
            block.append(f"  ## Step {j}: {step}")
            for k, st in enumerate(step_item.get("subtasks", []), start=1):
                subtask = st.get("subtask", "")
                block.append(f"     - Subtask {j}.{k}: {subtask}")
        lines.append("\n".join(block))
    return "\n\n".join(lines).strip()


def build_plan_subtask_action_examples(
    similars: List[Any], max_items: int = 3
) -> str:
    """Build examples showing step → subtask_action structure from plan_subtask_action."""
    lines: List[str] = []
    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or ""
        items = d.get("plan_subtask_action") or []
        if not items:
            continue
        block = [f"[Similar Task #{i}] {task}"]
        for j, step_item in enumerate(items, start=1):
            step = step_item.get("step", "")
            block.append(f"  ## Step {j}: {step}")
            for k, st in enumerate(step_item.get("subtasks", []), start=1):
                subtask_action = st.get("subtask_action", "")
                block.append(f"     - Subtask {j}.{k}: {subtask_action}")
        lines.append("\n".join(block))
    return "\n\n".join(lines).strip()



def build_similar_task_direction_blocks(similars: List[Any], max_items: int = 3) -> str:
    lines: List[str] = []

    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or ""
        knowledge = (d.get("task_analysis") or {}).get("knowledge", "")
        lines.append(f"[Similar Task #{i}] {task}\nKnowledge: {knowledge}\n")
    return "\n\n".join(lines).strip()


def build_similar_task_plan_blocks(similars: List[Any], max_items: int = 3) -> str:
    """Build example blocks showing only the plan steps from task_analysis."""
    lines: List[str] = []

    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or ""
        plan_steps = (d.get("task_analysis") or {}).get("plan") or []
        plan_str = "\n".join(f"  - {s}" for s in plan_steps) if plan_steps else ""
        lines.append(f"[Similar Task #{i}] {task}\nPlan:\n{plan_str}\n")
    return "\n\n".join(lines).strip()


def generate_knowledge(
    task: str,
    examples: str,
    planning_prompt_template: Dict[str, str],
    model_name: str,
    key: str,
    url: str,
    model: Any,
    slm: bool,
) -> str:
    """Stage 1: Generate domain knowledge (declarative + procedural) for the given task."""
    prompt = populate_template(
        planning_prompt_template["knowledge_prompt"],
        variables={"task": task, "examples": examples},
    )
    knowledge_str = call_model(
        query=prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"[Stage 1] Knowledge Prompt:\n{prompt}")
    logger.info(f"[Stage 1] Generated Knowledge:\n{knowledge_str}")
    logger.info("=" * 100)
    return knowledge_str


def generate_plan(
    task: str,
    knowledge: str,
    examples: str,
    planning_prompt_template: Dict[str, str],
    model_name: str,
    key: str,
    url: str,
    model: Any,
    slm: bool,
) -> str:
    """Stage 2: Generate step-by-step plan from task and knowledge."""
    prompt = populate_template(
        planning_prompt_template["plan_prompt"],
        variables={"task": task, "knowledge": knowledge, "examples": examples},
    )
    plan_str = call_model(
        query=prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"[Stage 2] Plan Prompt:\n{prompt}")
    logger.info(f"[Stage 2] Generated Plan:\n{plan_str}")
    logger.info("=" * 100)
    return plan_str


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
    plan_mode: Optional[str] = None,
    tools=None,
    managed_agents=None,
    planning_prompt_templates=None,
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
    # ====== Log query task_type/domain and retrieved task_type/domain ====== #
    if retrieval_option == "type_and_domain" and type_domain_retrieval_method is not None:
        logger.info(f"[Retrieval Query] task_type={task_types}, domain={domains}")
    else:
        logger.info("[Retrieval Query] task_type=N/A (text-only retrieval), domain=N/A")

    for i, r in enumerate(retrieval_results):
        ta = r.get("task_analysis") or {}
        r_types = ta.get("task_type", [])
        r_domains = ta.get("domain", [])
        logger.info(f"[Retrieved #{i+1}] task_id={r.get('task_id')}, task_type={r_types}, domain={r_domains}")

    knowledge_examples = build_similar_task_direction_blocks(similars=retrieval_results)
    plan_examples = build_similar_task_plan_blocks(similars=retrieval_results)
    logger.info(f"Retrieved knowledge examples:\n {knowledge_examples}")
    logger.info(f"Retrieved plan examples:\n {plan_examples}")

    # ====== [2] Stage 1: Generate knowledge ====== #
    knowledge_text = generate_knowledge(
        task=augmented_question,
        examples=knowledge_examples,
        planning_prompt_template=planning_prompt_template,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )

    # ====== [3] Stage 2: Generate plan ====== #
    plan_str = generate_plan(
        task=augmented_question,
        knowledge=knowledge_text,
        examples=plan_examples,  # plan steps only
        planning_prompt_template=planning_prompt_template,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )

    examples = {
        "plan_subtask": build_plan_subtask_examples(retrieval_results),
        "plan_subtask_action": build_plan_subtask_action_examples(retrieval_results),
    }

    # ====== [4] Stage 4: generate subtasks from plan_str (subtask / plan_subtask / plan_subtask_action modes) ====== #
    if plan_mode in ("subtask", "plan_subtask", "plan_subtask_action") and planning_prompt_templates:
        _tools = tools or {}
        _managed_agents = managed_agents or {}

        example_key = "plan_subtask_action" if plan_mode == "plan_subtask_action" else "plan_subtask"
        stage2_prompt = populate_template(
            planning_prompt_templates["initial_plan_subtask_action_stage2"],
            variables={
                "task": augmented_question,
                "tools": _tools,
                "managed_agents": _managed_agents,
                "plan_steps": plan_str,
                "examples": examples[example_key],
            },
        )
        subtask_plan = call_model(
            query=stage2_prompt,
            model_name=model_name,
            key=key,
            url=url,
            model=model,
            slm=slm,
        )
        logger.info("=" * 100)
        logger.info(f"[Stage 4] Subtask Prompt:\n{stage2_prompt}")
        logger.info(f"[Stage 4] Subtask Plan:\n{subtask_plan}")
        logger.info("=" * 100)

        final_plan = subtask_plan if plan_mode == "subtask" else f"Plan:\n{plan_str}\n\nSubtasks:\n{subtask_plan}"
    else:
        final_plan = plan_str

    return {
        "plan": final_plan,
        "knowledge": knowledge_text,
        "retrieval_results": retrieval_results,
        "examples": examples,
    }
