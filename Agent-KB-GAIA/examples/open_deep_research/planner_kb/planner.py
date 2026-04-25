from __future__ import annotations

import json
import os

from smolagents.agents import populate_template
from agent_kb.agent_kb_utils import call_model, BUAKBClient
from typing import Any, Dict, List, Literal, Optional

from .mece_utils import load_prompts

# planner_prompts.yaml 의 상대경로 (절대경로 하드코딩 제거)
_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planner_prompts.yaml")

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Predefined constants for type_and_domain retrieval mode.
# These map to task_analysis.task_type and task_analysis.domain in the KB.
# Modify as needed.
# ---------------------------------------------------------------------------
TASK_TYPE_CONSTANTS: List[str] = ['algebra', 'algorithm', 'analysis', 'approximation', 'calculation', 'computation', 'decision_support', 'engineering', 'geometry', 'simulations']

DOMAIN_CONSTANTS: List[str] = ['arts_humanities', 'business', 'computer_science', 'engineering', 'law', 'mathematics', 'medicine', 'science', 'social_sciences', 'technology']

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
        plan_raw = (d.get("task_analysis") or {}).get("plan") or []
        if isinstance(plan_raw, list):
            plan_str = "\n".join(f"  - {s}" for s in plan_raw)
        else:
            plan_str = str(plan_raw)
        lines.append(f"[Similar Task #{i}] {task}\nPlan:\n{plan_str}\n")
    return "\n\n".join(lines).strip()


def build_instance_examples(similars: List[Any], max_items: int = 3) -> str:
    """Build example blocks showing concrete execution instances."""
    lines: List[str] = []
    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or ""
        instance = d.get("instance") or (d.get("task_analysis") or {}).get("instance") or ""
        if instance:
            lines.append(f"[Similar Task #{i}] {task}\nInstance:\n{instance}\n")
    return "\n\n".join(lines).strip()


def build_decision_guide_blocks(similars: List[Any], level: Optional[str] = None, max_items: int = 3) -> str:
    """Build Decision Guide (G/S) blocks from signals_summary of retrieved tasks.

    Each guide entry has: level, failures/failure, causes/cause, corrections/correction
    level filter: 'knowledge' | 'plan' | 'instance' | None (all levels)
    """
    lines: List[str] = []
    count = 0
    for d in similars:
        if count >= max_items:
            break
        task = d.get("task") or ""
        guides = d.get("decision_guide") or []
        if not guides:
            continue
        task_lines = [f"[From Task: {task[:80]}]"]
        for g in guides:
            g_level = g.get("level", "")
            if level and g_level != level:
                continue
            failures = g.get("failures") or ([g.get("failure")] if g.get("failure") else [])
            causes = g.get("causes") or ([g.get("cause")] if g.get("cause") else [])
            corrections = g.get("corrections") or ([g.get("correction")] if g.get("correction") else [])
            for fail, cause, corr in zip(
                failures or ["(unknown)"],
                causes or ["(unknown)"],
                corrections or ["(unknown)"],
            ):
                task_lines.append(
                    f"  [{g_level.upper()}] Failure: {fail}\n"
                    f"           Cause: {cause}\n"
                    f"           Correction: {corr}"
                )
        if len(task_lines) > 1:
            lines.append("\n".join(task_lines))
            count += 1
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
    decision_guide: str = "",
) -> str:
    """Stage 1: Generate domain knowledge (declarative + procedural) for the given task."""
    prompt = populate_template(
        planning_prompt_template["knowledge_prompt"],
        variables={"task": task, "examples": examples, "decision_guide": decision_guide},
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
    decision_guide: str = "",
) -> str:
    """Stage 2: Generate step-by-step plan from task and knowledge."""
    prompt = populate_template(
        planning_prompt_template["plan_prompt"],
        variables={"task": task, "knowledge": knowledge, "examples": examples, "decision_guide": decision_guide},
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


def generate_instance(
    task: str,
    knowledge: str,
    plan: str,
    examples: str,
    planning_prompt_template: Dict[str, str],
    model_name: str,
    key: str,
    url: str,
    model: Any,
    slm: bool,
    decision_guide: str = "",
) -> str:
    """Stage 3: Generate a concrete execution instance grounding the plan."""
    prompt = populate_template(
        planning_prompt_template["instance_prompt"],
        variables={
            "task": task,
            "knowledge": knowledge,
            "plan": plan,
            "examples": examples,
            "decision_guide": decision_guide,
        },
    )
    instance_str = call_model(
        query=prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"[Stage 3] Instance Prompt:\n{prompt}")
    logger.info(f"[Stage 3] Generated Instance:\n{instance_str}")
    logger.info("=" * 100)
    return instance_str


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


def plan_mode_planning(
    example: dict,
    retrieval_method,
    top_k: int,
    plan_mode: str,
) -> str:
    """
    Simple static injection using KB-retrieved similar tasks.

    Retrieves similar tasks and builds an additional_knowledge string
    incorporating knowledge + plan/subtask/action levels based on plan_mode.
    No LLM call is made — raw KB data is directly injected as a prefix.

    plan_mode values:
      - "plan"                : knowledge + plan steps
      - "subtask"             : knowledge + subtasks
      - "plan_subtask"        : knowledge + plan steps + subtasks
      - "plan_subtask_action" : knowledge + plan steps + subtask actions
    """
    results = retrieval_method(example["question"], top_k=top_k)
    if not results:
        return ""

    blocks = []
    for i, item in enumerate(results, 1):
        task = item.get("task", "")
        ta = item.get("task_analysis") or {}
        block_lines = [f"[Similar Task #{i}] {task}"]

        # Always include knowledge
        knowledge = ta.get("knowledge", "")
        if knowledge:
            block_lines.append(f"Knowledge: {knowledge}")

        # Include plan steps
        if plan_mode in ("plan", "plan_subtask", "plan_subtask_action"):
            plan_raw = ta.get("plan") or []
            if plan_raw:
                plan_str = (
                    "\n".join(f"  - {s}" for s in plan_raw)
                    if isinstance(plan_raw, list)
                    else str(plan_raw)
                )
                block_lines.append(f"Plan:\n{plan_str}")

        # Include instance (grounded execution example)
        instance = item.get("instance") or ta.get("instance") or ""
        if instance:
            block_lines.append(f"Instance:\n{instance}")

        # Include decision guide signals
        guides = item.get("decision_guide") or []
        if guides:
            guide_lines = ["Decision Guide:"]
            for g in guides:
                failures = g.get("failures") or ([g.get("failure")] if g.get("failure") else [])
                causes = g.get("causes") or ([g.get("cause")] if g.get("cause") else [])
                corrections = g.get("corrections") or ([g.get("correction")] if g.get("correction") else [])
                for fail, cause, corr in zip(
                    failures or [""], causes or [""], corrections or [""]
                ):
                    guide_lines.append(
                        f"  [{g.get('level','').upper()}] Failure: {fail} | Cause: {cause} | Correction: {corr}"
                    )
            block_lines.append("\n".join(guide_lines))

        # Include subtasks or subtask-actions (legacy)
        if plan_mode in ("subtask", "plan_subtask", "plan_subtask_action"):
            psa_items = item.get("plan_subtask_action") or []
            if psa_items:
                detail_lines = []
                for j, step_item in enumerate(psa_items, 1):
                    step = step_item.get("step", "")
                    detail_lines.append(f"  ## Step {j}: {step}")
                    for k, st in enumerate(step_item.get("subtasks", []), 1):
                        if plan_mode == "plan_subtask_action":
                            text = st.get("subtask_action", "") or st.get("subtask", "")
                        else:
                            text = st.get("subtask", "")
                        detail_lines.append(f"     - {j}.{k}: {text}")
                label = "Subtask Actions" if plan_mode == "plan_subtask_action" else "Subtasks"
                block_lines.append(f"{label}:\n" + "\n".join(detail_lines))

        blocks.append("\n".join(block_lines))

    return "Here are similar task examples for reference:\n\n" + "\n\n".join(blocks)


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
        path=_PROMPT_PATH
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
    instance_examples = build_instance_examples(similars=retrieval_results)
    guide_knowledge = build_decision_guide_blocks(retrieval_results, level="knowledge")
    guide_plan = build_decision_guide_blocks(retrieval_results, level="plan")
    guide_instance = build_decision_guide_blocks(retrieval_results, level="instance")
    logger.info(f"Retrieved knowledge examples:\n {knowledge_examples}")
    logger.info(f"Retrieved plan examples:\n {plan_examples}")
    logger.info(f"Retrieved instance examples:\n {instance_examples}")

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
        decision_guide=guide_knowledge,
    )

    # ====== [3] Stage 2: Generate plan ====== #
    plan_str = generate_plan(
        task=augmented_question,
        knowledge=knowledge_text,
        examples=plan_examples,
        planning_prompt_template=planning_prompt_template,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
        decision_guide=guide_plan,
    )

    # ====== [4] Stage 3: Generate instance (concrete execution grounding) ====== #
    instance_str = generate_instance(
        task=augmented_question,
        knowledge=knowledge_text,
        plan=plan_str,
        examples=instance_examples,
        planning_prompt_template=planning_prompt_template,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
        decision_guide=guide_instance,
    )

    subtask_examples = {
        "plan_subtask": build_plan_subtask_examples(retrieval_results),
        "plan_subtask_action": build_plan_subtask_action_examples(retrieval_results),
    }

    # ====== [5] Stage 4: generate subtasks (subtask / plan_subtask / plan_subtask_action modes) ====== #
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
                "examples": subtask_examples[example_key],
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
        logger.info(f"[Stage 5] Subtask Prompt:\n{stage2_prompt}")
        logger.info(f"[Stage 5] Subtask Plan:\n{subtask_plan}")
        logger.info("=" * 100)

        if plan_mode == "subtask":
            final_plan = subtask_plan
        else:
            final_plan = f"Knowledge:\n{knowledge_text}\n\nPlan:\n{plan_str}\n\nInstance:\n{instance_str}\n\nSubtasks:\n{subtask_plan}"
    else:
        final_plan = f"Knowledge:\n{knowledge_text}\n\nPlan:\n{plan_str}\n\nInstance:\n{instance_str}"

    return {
        "plan": final_plan,
        "knowledge": knowledge_text,
        "plan_steps": plan_str,
        "instance": instance_str,
        "retrieval_results": retrieval_results,
        "examples": subtask_examples,
    }


# ---------------------------------------------------------------------------
# Dynamic proposal planning (retrieval_mode=td, plan_mode=dynamic)
# ---------------------------------------------------------------------------

_MAPPING_CACHE: Optional[Dict] = None
_MAPPING_PATH_DEFAULT = os.path.join(
    os.path.dirname(__file__), "..", "agent_kb", "mapping.json"
)


def _load_mapping(mapping_path: Optional[str] = None) -> Dict:
    global _MAPPING_CACHE
    if _MAPPING_CACHE is None:
        path = mapping_path or _MAPPING_PATH_DEFAULT
        try:
            with open(os.path.abspath(path), "r", encoding="utf-8") as f:
                data = json.load(f)
            _MAPPING_CACHE = data.get("entries", data)
        except Exception as e:
            logger.warning(f"Failed to load mapping.json from {path}: {e}")
            _MAPPING_CACHE = {}
    return _MAPPING_CACHE


def _lookup_mapping(
    type_task_path: str,
    domain_path: str,
    mapping: Dict,
) -> Dict[str, Any]:
    """Look up mapping config for (type_task_path, domain_path).

    Fallback order:
      1. exact "{type_task_path}|{domain_path}"
      2. "{type_task_path}" (type only, full path)
      3. "{domain_path}" (domain only, full path)
      4. parent of type_task_path (level1/level2)
      5. parent of domain_path (level1/level2)
      6. level1 of type_task_path
      7. level1 of domain_path
      8. "default"
    """
    type_levels = [p for p in type_task_path.split("/") if p]
    domain_levels = [p for p in domain_path.split("/") if p]

    candidates = [
        f"{type_task_path}|{domain_path}",
        type_task_path,
        domain_path,
    ]
    # type parent paths (level1/level2, level1)
    for n in range(len(type_levels) - 1, 0, -1):
        candidates.append("/".join(type_levels[:n]))
    # domain parent paths (level1/level2, level1)
    for n in range(len(domain_levels) - 1, 0, -1):
        candidates.append("/".join(domain_levels[:n]))
    candidates.append("default")

    for key in candidates:
        if key in mapping:
            logger.info(f"[Mapping] matched key='{key}' for type='{type_task_path}', domain='{domain_path}'")
            return mapping[key]

    return {"fields": ["knowledge", "plan"], "depth": "plan_subtask"}


def classify_task_type_domain_path(
    task: str,
    planning_prompt_template: Dict[str, str],
    model_name: str,
    key: str,
    url: str,
    model: Any,
    slm: bool,
) -> Dict[str, str]:
    """Classify task into 3-level type_task_path and domain_path using LLM."""
    prompt = populate_template(
        planning_prompt_template["task_classification_path_prompt"],
        variables={"task": task},
    )
    raw = call_model(
        query=prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        type_task_path = result.get("type_task_path", "problem_solving/general/general")
        domain_path = result.get("domain_path", "science/general/general")
    except Exception as e:
        logger.warning(f"Path classification parsing failed: {e}. Raw: {raw}")
        type_task_path = "problem_solving/general/general"
        domain_path = "science/general/general"

    logger.info(f"[Path Classification] type_task_path={type_task_path}, domain_path={domain_path}")
    return {"type_task_path": type_task_path, "domain_path": domain_path}


def build_dynamic_reference_blocks(
    similars: List[Any],
    fields: List[str],
    depth: str,
    max_items: int = 3,
) -> str:
    """Build reference document blocks from similar tasks using mapping-specified fields and depth.

    fields: subset of ["knowledge", "plan", "agent_experience", "subtask_action"]
    depth:  "plan" | "plan_subtask" | "plan_subtask_action"
    """
    lines: List[str] = []

    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or ""
        ta = d.get("task_analysis") or {}
        block_lines = [f"[Similar Task #{i}] {task}"]

        if "knowledge" in fields:
            knowledge = ta.get("knowledge", "")
            if knowledge:
                block_lines.append(f"Knowledge: {knowledge}")

        if "agent_experience" in fields:
            exp = d.get("agent_experience", "")
            if exp:
                block_lines.append(f"Experience: {exp}")

        if "plan" in fields or depth in ("plan", "plan_subtask", "plan_subtask_action"):
            plan_steps = ta.get("plan") or []
            if plan_steps:
                plan_str = "\n".join(f"  - {s}" for s in plan_steps)
                block_lines.append(f"Plan:\n{plan_str}")

        if depth in ("plan_subtask", "plan_subtask_action"):
            psa_items = d.get("plan_subtask_action") or []
            if psa_items:
                detail_lines = []
                for j, step_item in enumerate(psa_items, 1):
                    step = step_item.get("step", "")
                    detail_lines.append(f"  ## Step {j}: {step}")
                    for k, st in enumerate(step_item.get("subtasks", []), 1):
                        if depth == "plan_subtask_action":
                            text = st.get("subtask_action", "") or st.get("subtask", "")
                        else:
                            text = st.get("subtask", "")
                        detail_lines.append(f"     - {j}.{k}: {text}")
                label = "Subtask Actions" if depth == "plan_subtask_action" else "Subtasks"
                block_lines.append(f"{label}:\n" + "\n".join(detail_lines))

        lines.append("\n".join(block_lines))

    return "\n\n".join(lines).strip()


def dynamic_proposal_planning(
    example: dict,
    augmented_question: str,
    model_name: str,
    key: str,
    url: str,
    model: Any,
    slm: bool,
    td_retrieval_method,
    retrieval_method,
    top_k: int,
    mapping_path: Optional[str] = None,
    tools=None,
    managed_agents=None,
    planning_prompt_templates=None,
):
    """Dynamic variant of proposal_planning driven by mapping.json.

    Steps:
      1. Classify task into type_task_path / domain_path (3-level paths).
      2. Retrieve similar tasks via td_retrieval_method (retrieval_mode=td).
         Falls back to retrieval_method if td returns nothing.
      3. Look up mapping.json for (type_task_path, domain_path) to get fields + depth.
      4. Build reference doc blocks using the mapped fields and depth.
      5. Stage 1 — generate knowledge (same as proposal_planning).
      6. Stage 2 — generate plan using mapping-depth reference examples.
      7. Optionally generate subtask breakdown (plan_mode=dynamic maps depth to stage4).
    """
    planning_prompt_template = load_prompts(
        path=_PROMPT_PATH
    )

    # ====== [1] Classify task → 3-level paths ====== #
    classification = classify_task_type_domain_path(
        task=augmented_question,
        planning_prompt_template=planning_prompt_template,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    type_task_path = classification["type_task_path"]
    domain_path = classification["domain_path"]

    # ====== [2] Retrieve with td (path-filtered hybrid) ====== #
    retrieval_results = td_retrieval_method(
        example["question"],
        type_task_path=type_task_path,
        domain_path=domain_path,
        top_k=top_k,
    )
    if not retrieval_results:
        logger.warning("td_hybrid_search returned no results; falling back to hybrid retrieval.")
        retrieval_results = retrieval_method(example["question"], top_k=top_k)

    for i, r in enumerate(retrieval_results):
        ta = r.get("task_analysis") or {}
        logger.info(
            f"[Retrieved #{i+1}] task_id={r.get('task_id')}, "
            f"type={ta.get('task_type_normalized')}, domain={ta.get('domain_normalized')}, "
            f"score={r.get('score', 0):.4f}"
        )

    # ====== [3] Load mapping.json → fields + depth ====== #
    mapping = _load_mapping(mapping_path)
    config = _lookup_mapping(type_task_path, domain_path, mapping)
    fields = config.get("fields", ["knowledge", "plan"])
    depth = config.get("depth", "plan_subtask")
    logger.info(f"[Mapping Config] fields={fields}, depth={depth}")

    # ====== [4] Build reference blocks ====== #
    knowledge_examples = build_dynamic_reference_blocks(
        similars=retrieval_results,
        fields=[f for f in fields if f in ("knowledge", "agent_experience")],
        depth="plan",  # knowledge stage only needs plan structure at most
        max_items=3,
    )
    plan_examples = build_dynamic_reference_blocks(
        similars=retrieval_results,
        fields=fields,
        depth=depth,
        max_items=3,
    )
    logger.info(f"[Dynamic] Knowledge examples:\n{knowledge_examples}")
    logger.info(f"[Dynamic] Plan examples:\n{plan_examples}")

    # ====== [5] Stage 1: Generate knowledge ====== #
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

    # ====== [6] Stage 2: Generate plan ====== #
    plan_str = generate_plan(
        task=augmented_question,
        knowledge=knowledge_text,
        examples=plan_examples,
        planning_prompt_template=planning_prompt_template,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )

    # ====== [7] Stage 4 (optional): Generate subtasks based on depth ====== #
    if depth in ("plan_subtask", "plan_subtask_action") and planning_prompt_templates:
        _tools = tools or {}
        _managed_agents = managed_agents or {}

        example_key = "plan_subtask_action" if depth == "plan_subtask_action" else "plan_subtask"
        subtask_examples = {
            "plan_subtask": build_plan_subtask_examples(retrieval_results),
            "plan_subtask_action": build_plan_subtask_action_examples(retrieval_results),
        }
        stage2_prompt = populate_template(
            planning_prompt_templates["initial_plan_subtask_action_stage2"],
            variables={
                "task": augmented_question,
                "tools": _tools,
                "managed_agents": _managed_agents,
                "plan_steps": plan_str,
                "examples": subtask_examples[example_key],
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
        logger.info(f"[Dynamic Stage 4] Subtask Plan:\n{subtask_plan}")
        logger.info("=" * 100)

        final_plan = subtask_plan if depth == "plan_subtask" else f"Plan:\n{plan_str}\n\nSubtasks:\n{subtask_plan}"
    else:
        final_plan = plan_str

    return {
        "plan": final_plan,
        "knowledge": knowledge_text,
        "retrieval_results": retrieval_results,
        "type_task_path": type_task_path,
        "domain_path": domain_path,
        "mapping_config": config,
    }


# ---------------------------------------------------------------------------
# Bu-taxonomy dynamic proposal planning
# ---------------------------------------------------------------------------

def _format_bu_taxonomy_results(results: List[Dict], max_items: int = 3) -> str:
    """Bu-taxonomy 검색 결과의 dynamic_proposal을 additional_knowledge 문자열로 변환."""
    blocks: List[str] = []
    for i, item in enumerate(results[:max_items], start=1):
        task    = item.get("task", "")
        proposal = item.get("dynamic_proposal") or {}
        lines   = [f"[Similar Task #{i}] {task}"]

        knowledge = proposal.get("knowledge")
        if knowledge:
            lines.append(f"Knowledge: {knowledge}")

        plan = proposal.get("plan")
        if plan:
            if isinstance(plan, list):
                plan_str = "\n".join(f"  - {s}" for s in plan)
            else:
                plan_str = str(plan)
            lines.append(f"Plan:\n{plan_str}")

        subtasks = proposal.get("subtasks")
        if subtasks:
            lines.append("Subtasks:")
            for s in subtasks:
                lines.append(f"  - {s}")

        actions = proposal.get("actions")
        if actions:
            detail: List[str] = []
            for j, step_item in enumerate(actions, 1):
                step = step_item.get("step", "")
                detail.append(f"  ## Step {j}: {step}")
                for k, st in enumerate(step_item.get("subtasks", []), 1):
                    text = st.get("subtask_action", "") or st.get("subtask", "")
                    detail.append(f"     - {j}.{k}: {text}")
            lines.append("Subtask Actions:\n" + "\n".join(detail))

        blocks.append("\n".join(lines))

    if not blocks:
        return ""
    return "Here are similar task examples for reference (bu-taxonomy):\n\n" + "\n\n".join(blocks)


def bu_dynamic_proposal_planning(
    example: dict,
    top_k: int = 3,
    min_pool_size: int = 20,
    hybrid_weights: Optional[Dict[str, float]] = None,
    bu_client: Optional[Any] = None,
    model_name: Optional[str] = None,
    key: Optional[str] = None,
    url: Optional[str] = None,
    model: Optional[Any] = None,
    slm: bool = False,
    force_depth: Optional[str] = None,
) -> str:
    """Bu-taxonomy KB 기반 dynamic proposal planning (2-stage generation).

    Steps:
      1. BUAKBClient.taxonomy_search() — LLM 계층 분류 + 필터링 + hybrid 리랭킹 + dynamic_proposal
      2. mapping table 기반 config 확인:
         - recommended_knowledge=True → Stage 1: LLM으로 knowledge 생성
         - recommended_depth 에 따라 → Stage 2: LLM으로 plan 생성

    model_name/key/url/model/slm: LLM generation 에 사용 (None이면 formatted raw 결과 반환)
    """
    client   = bu_client or BUAKBClient()
    question = example.get("question") or example.get("task", "")

    results = client.taxonomy_search(
        query            = question,
        top_k            = top_k,
        min_pool_size    = min_pool_size,
        hybrid_weights   = hybrid_weights or {"text": 0.5, "semantic": 0.5},
        include_proposal = True,
    )

    if not results:
        logger.warning("[bu_dynamic_proposal_planning] taxonomy_search returned no results.")
        return ""

    # ── 로그: 조회 결과 ──────────────────────────────────────────────────
    for i, r in enumerate(results):
        bp           = r.get("bu_taxonomy_path") or {}
        tt           = (bp.get("task_type") or {}).get("minor_label", "N/A")
        dm           = (bp.get("domain")    or {}).get("minor_label", "N/A")
        proposal_cfg = (r.get("dynamic_proposal") or {}).get("_config", {})
        logger.info(
            f"[BU Retrieved #{i+1}] task_id={r.get('task_id')}, "
            f"task_type={tt}, domain={dm}, "
            f"depth={proposal_cfg.get('recommended_depth')}, "
            f"confidence={proposal_cfg.get('confidence')}, "
            f"score={r.get('total_score', 0):.4f}"
        )

    # ── generation 불가능하면 raw 포맷 반환 ──────────────────────────────
    if not model_name and not slm:
        additional_knowledge = _format_bu_taxonomy_results(results)
        logger.info(f"[bu_dynamic_proposal_planning] raw format:\n{additional_knowledge}")
        return additional_knowledge

    # ── config 추출 (첫 번째 결과 기준) ─────────────────────────────────
    first_cfg      = (results[0].get("dynamic_proposal") or {}).get("_config", {})
    need_knowledge = first_cfg.get("recommended_knowledge", True)
    depth          = force_depth or first_cfg.get("recommended_depth", "plan_subtask")
    logger.info(f"[bu_dynamic_proposal_planning] depth={depth} (force={force_depth!r})")

    planning_prompt_template = load_prompts(path=_PROMPT_PATH)

    # ── Stage 1: knowledge 생성 (조건부) ─────────────────────────────────
    knowledge_text = ""
    if need_knowledge:
        knowledge_examples = build_similar_task_direction_blocks(similars=results)
        knowledge_text = generate_knowledge(
            task                     = question,
            examples                 = knowledge_examples,
            planning_prompt_template = planning_prompt_template,
            model_name               = model_name,
            key                      = key,
            url                      = url,
            model                    = model,
            slm                      = slm,
        )

    # ── Stage 2: plan 생성 (depth 기반 예시 선택) ────────────────────────
    if depth == "full":
        plan_examples = build_plan_subtask_action_examples(similars=results)
    elif depth == "plan_subtask":
        plan_examples = build_plan_subtask_examples(similars=results)
    else:  # plan_only
        plan_examples = build_similar_task_plan_blocks(similars=results)

    plan_text = generate_plan(
        task                     = question,
        knowledge                = knowledge_text,
        examples                 = plan_examples,
        planning_prompt_template = planning_prompt_template,
        model_name               = model_name,
        key                      = key,
        url                      = url,
        model                    = model,
        slm                      = slm,
    )

    # ── 최종 포맷 ─────────────────────────────────────────────────────────
    parts: List[str] = ["[BU-KB Guidance for Current Task]"]
    if knowledge_text:
        parts.append(f"Knowledge:\n{knowledge_text}")
    if plan_text:
        parts.append(f"Plan:\n{plan_text}")

    additional_knowledge = "\n\n".join(parts)
    logger.info(f"[bu_dynamic_proposal_planning] generated:\n{additional_knowledge}")
    return additional_knowledge
