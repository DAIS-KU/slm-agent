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


def build_similar_task_blocks(
    similars: List[Any], mode, max_items: int = 5, use_summary=False
) -> str:
    lines: List[str] = []
    # logger.info(f"build_similar_task_blocks example: {similars[0]}")

    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or d.get("query") or d.get("question") or ""
        domain = d.get("domain")
        skills = d.get("skills")
        objective = d.get("objective")
        knowledge = d.get("knowledge_summary") if use_summary else d.get("knowledge")
        constraints = (
            d.get("constraints_summary") if use_summary else d.get("constraints")
        )
        instructions = (
            d.get("instructions_summary") if use_summary else d.get("instructions")
        )
        approach = d.get("approach_summary") if use_summary else d.get("approach")
        plan = d.get("plan_summary") if use_summary else d.get("plan")
        agent_planning = d.get("agent_planning")

        parts: List[str] = []
        if mode == "knowledge":
            parts.append(f"[Similar Task #{i}] {task}")
            parts.append(f"Knowledge: {knowledge}\n")
        elif mode == "ci":
            parts.append(f"[Similar Task #{i}] {task}")
            parts.append(f"Constraints: {constraints}")
            parts.append(f"Instructions: {constraints}\n")
        elif mode == "approach":
            parts.append(f"[Similar Task #{i}] {task}")
            parts.append(f"Approach:{approach}")
        elif mode == "plan":
            parts.append(f"[Similar Task #{i}] {task}")
            parts.append(f"Plan: {plan}\n")
        elif mode == "spec":
            parts.append(
                f"[Similar Task #{i}] {task}\nDomain:{domain}\nSkills:{skills}\nObjective:{objective}\n"
            )
        elif mode == "agent_planning":
            parts.append(f"[Similar Task #{i}] {task}")
            parts.append(f"Plan: {agent_planning}\n")
        lines.append("\n".join(parts).strip())

    return lines if mode == "spec" else "\n\n".join(lines).strip()


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
    planning_field="plan",
    use_summary=False,
):
    planning_prompt_template = load_prompts(
        path="/home/huijeong/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/planner_prompts.yaml"
    )
    if retrieval_method is None:
        logger.info(f"planning_task - retrieval_method is None.")
        planning_prompt = populate_template(
            planning_prompt_template["planning_prompt"],
            variables={"task": augmented_question},
        )
    else:
        logger.info(f"planning_task - retrieval_method is not None.")
        retrieval_results = retrieval_method(example["question"], top_k=top_k)
        # logger.info(f"Retrieved retrieval_results:\n {retrieval_results}")
        examples = build_similar_task_blocks(
            similars=retrieval_results, mode=planning_field, use_summary=use_summary
        )
        logger.info(f"Retrieved examples:\n {examples}")
        planning_prompt = populate_template(
            planning_prompt_template["planning_with_examples_prompt"],
            variables={
                "task": augmented_question,
                "examples": examples,
            },
        )
    plan_str = call_model(
        query=planning_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"Generated Plan:\n{plan_str}")
    logger.info("=" * 100)

    progressive_ci_prompt_template = planning_prompt_template["progressive_ci_prompt"]
    similar_blocks_ci = build_similar_task_blocks(
        retrieval_results, mode="ci", use_summary=use_summary
    )
    progressive_ci_prompt = populate_template(
        progressive_ci_prompt_template,
        variables={"task": augmented_question, "similar_blocks": similar_blocks_ci},
    )
    ci_str = call_model(
        query=progressive_ci_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    # ci_str = call_model(
    #     query=f"Summerize the following text.\n\n{raw_ci_str}",
    #     model_name=model_name,
    #     key=key,
    #     url=url,
    #     model=model,
    #     slm=slm,
    # )
    logger.info("=" * 100)
    logger.info(f"Generated Constraitns/Instructions:\n{ci_str}")
    logger.info("=" * 100)
    return plan_str, ci_str


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
    logger.info("=" * 100)
    logger.info("Start to generate knowledge")
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
    knowledge_str = call_model(
        query=progressive_knowledge_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    # knowledge_str = call_model(
    #     query=f"Summerize the following text.\nTEXT:\n\n{raw_knowledge_str}",
    #     model_name=model_name,
    #     key=key,
    #     url=url,
    #     model=model,
    #     slm=slm,
    # )
    logger.info(f"Generated Knowledge Prompt:\n{progressive_knowledge_prompt}")
    # logger.info(f"Generated Raw Knowledge:\n{raw_knowledge_str}")
    logger.info(f"Generated Summarized Knowledge:\n{knowledge_str}")
    logger.info("=" * 100)

    # 2) constraints, instructions
    logger.info("=" * 100)
    logger.info("Start to generate constraints and instructions")
    progressive_ci_prompt_template = planning_prompt_template["progressive_ci_prompt"]
    similar_blocks_ci = build_similar_task_blocks(
        retrieval_results, mode="ci", use_summary=use_summary
    )
    progressive_ci_prompt = populate_template(
        progressive_ci_prompt_template,
        variables={"task": augmented_question, "similar_blocks": similar_blocks_ci},
    )
    ci_str = call_model(
        query=progressive_ci_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    # ci_str = call_model(
    #     query=f"Summerize the following text.\nTEXT:\n\n{raw_ci_str}",
    #     model_name=model_name,
    #     key=key,
    #     url=url,
    #     model=model,
    #     slm=slm,
    # )
    logger.info(f"Generated Constraitns/Instructions Prompt:\n{progressive_ci_prompt}")
    logger.info(f"Generated Constraitns/Instructions:\n{ci_str}")
    logger.info("=" * 100)

    # 3) approach
    logger.info("=" * 100)
    logger.info("Start to generate approach")
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
    approach_str = call_model(
        query=progressive_approach_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    # approach_str = call_model(
    #     query=f"Summerize the following text.\nTEXT:\n\n{raw_approach_str}",
    #     model_name=model_name,
    #     key=key,
    #     url=url,
    #     model=model,
    #     slm=slm,
    # )
    logger.info(f"Generated Approach Prompt:\n{progressive_approach_prompt}")
    # logger.info(f"Generated Raw Approach:\n{raw_approach_str}")
    logger.info(f"Generated Summerized Approach:\n{approach_str}")
    logger.info("=" * 100)

    # 3) plan
    logger.info("=" * 100)
    logger.info("Start to generate plan")
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
            "similar_blocks": similar_blocks_plan,
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
    use_summary=False,
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
    ci_prompt_template = prompts["progressive_ci_prompt"]
    ci_prompt = populate_template(
        ci_prompt_template,
        variables={"task": augmented_question, "similar_blocks": None},
    )
    ci_str = call_model(
        query=ci_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    # ci_str = call_model(
    #     query=f"Summerize the following text.\n\n{raw_ci_str}",
    #     model_name=model_name,
    #     key=key,
    #     url=url,
    #     model=model,
    #     slm=slm,
    # )
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
    logger.info(f"ref_specs #{len(ref_specs)}")
    logger.info("=" * 100)
    for idx, ref_spec in enumerate(ref_specs):
        transter_ref_prompt_template = prompts["transfer_ka_prompt"]
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
        logger.info(f"Referred Spec:\n{ref_spec}")
        logger.info(f"Generated Tranfer Item:\n{transfer}")
        transfers.append(transfer)
    logger.info("=" * 100)
    proposed_ka = "\n".join(transfers)

    # ---- (3) 최종 knowledge and approach 생성 ----
    final_ka_prompt_template = prompts["final_ka_prompt"]
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
    plan_prompt_template = prompts["recontextualized_plan_prompt"]
    plan_prompt = populate_template(
        plan_prompt_template,
        variables={
            "task": augmented_question,
            "knowledge_approach": final_ka,
            "constraints_instructions": ci_str,
        },
    )
    plan_str = call_model(
        query=plan_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"Generated Final Plan:\n{plan_str}")
    logger.info("=" * 100)

    return plan_str, ci_str


def _safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()

    # 1) 바로 파싱
    try:
        return json.loads(text)
    except Exception:
        logger.info(f"Failed to parse JSON from model output:\n{text}")
        pass

    # 2) 첫 '{' ~ 마지막 '}' 범위 파싱
    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        candidate = text[l : r + 1]
        return json.loads(candidate)


def build_do_blocks(similars: List[Any], max_items: int = 5, do_field="do_raw") -> str:
    lines: List[str] = []
    logger.info(f"build_do_blocks example: {similars[0]}")
    for i, d in enumerate(similars[:max_items], start=1):
        subtask = d.get("subtask")
        _do = d.get(do_field)
        expected_answer = d.get("expected_answer")
        actual_answer = d.get("actual_answer")
        lines.append(
            f"[Similar SubTask #{i}] {subtask}\nSolve: {_do}\nExpected Answer: {expected_answer}, Final Answer: {actual_answer}"
        )
    return "\n\n".join(lines).strip()


def generate_plan_subtasks(
    example,
    augmented_question,
    model_name,
    key,
    url,
    model,
    slm,
    retrieval_method,
    sub_retrieval_method,
    top_k,
    use_summary=False,
    use_sub_ex=False,
    do_field="do_raw",
):
    planning_prompt_template = load_prompts(
        path="/home/huijeong/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/planner_prompts.yaml"
    )
    retrieval_results = retrieval_method(example["question"], top_k=top_k)
    examples = build_similar_task_blocks(
        similars=retrieval_results, mode="plan", use_summary=use_summary
    )
    logger.info(f"Retrieved examples:\n {examples}")
    planning_prompt = populate_template(
        planning_prompt_template["planning_with_examples_prompt"],
        variables={
            "task": augmented_question,
            "examples": examples,
        },
    )
    plan_str = call_model(
        query=planning_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info("=" * 100)
    logger.info(f"Generated Plan Prompt:\n{planning_prompt}")
    logger.info(f"Generated Plan:\n{plan_str}")
    logger.info("=" * 100)

    # 2) Plan -> subtasks 생성(추출)
    if use_sub_ex:
        subtask_prompt = populate_template(
            planning_prompt_template["plan_to_subtasks_with_examples_prompt"],
            variables={
                "task": augmented_question,
                "plan": plan_str,
            },
        )
    else:
        subtask_prompt = populate_template(
            planning_prompt_template["plan_to_subtasks_prompt"],
            variables={
                "task": augmented_question,
                "plan": plan_str,
            },
        )
    subtask_str = call_model(
        query=subtask_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    subtasks = _safe_json_loads(subtask_str)
    logger.info("=" * 100)
    logger.info(f"Generated Subtasks Prompt:\n{subtask_prompt}")
    logger.info(f"Generated Subtasks:\n{subtasks}")
    logger.info("=" * 100)
    results: List[Tuple[str, Any]] = []

    logger.info("=" * 100)
    for s in subtasks:
        subtask_text = (s.get("subtask") if isinstance(s, dict) else str(s)) or ""
        inputs = s.get("inputs")
        procedure = s.get("procedure")
        expected_output = s.get("expected_output")
        subtask_text = subtask_text.strip()
        if not subtask_text:
            logger.info(f"subtask_text is blank! {s}")
            continue
        retrieved_examples = sub_retrieval_method(subtask_text, top_k)
        do_examples = build_do_blocks(similars=retrieved_examples, do_field=do_field)

        solve_prompt = populate_template(
            planning_prompt_template["solve_subtask_prompt"],
            variables={
                "subtask": subtask_text,
                "inputs": inputs,
                "procedure": procedure,
                "expected_output": expected_output,
                "do_examples": do_examples,
            },
        )
        solve_str = call_model(
            query=solve_prompt,
            model_name=model_name,
            key=key,
            url=url,
            model=model,
            slm=slm,
        )
        solved = _safe_json_loads(solve_str)
        output = solved.get("output", None)
        logger.info(f"Generated Subtask Output Prompt:\n{solve_prompt}")
        logger.info(f"Generated Output:\n{output}")
        results.append((subtask_text, output))
    logger.info("=" * 100)

    return plan_str, results


def build_similar_task_direction_blocks(similars: List[Any], max_items: int = 3) -> str:
    lines: List[str] = []
    # logger.info(f"build_similar_task_blocks example: {similars[0]}")

    for i, d in enumerate(similars[:max_items], start=1):
        task = d.get("task") or d.get("query") or d.get("question") or ""
        problem_type = d.get("task_spec").get("problem_type")
        decision_criterion = d.get("task_spec").get("decision_criterion")
        approach = d.get("task_spec").get("approach")
        pparts.append(
            f"[Similar Task #{i}] {task}\ProblemType:{problem_type}\WhatToDerieve:{decision_criterion}\Approach:{approach}\n"
        )
        lines.append("\n".join(parts).strip())
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
):
    planning_prompt_template = load_prompts(
        path="/home/huijeong/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/planner_prompts.yaml"
    )

    # ====== [1] Generate decision_criterion, approach ====== #
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
        query=planning_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    goal_approach = _safe_json_loads(approach_str)
    logger.info("=" * 100)
    logger.info(f"Generated Approach Prompt:\n{approach_prompt}")
    logger.info(f"Generated Approach:\n{goal_approach}")
    logger.info("=" * 100)

    # ====== [2] Generate plan ====== #
    approach_to_plan_prompt = populate_template(
        planning_prompt_template["approach_to_plan_prompt"],
        variables={
            "task": augmented_question,
            "decision_criterion": goal_approach.get("decision_criterion"),
            "approach": goal_approach.get("approach"),
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
    steps = _safe_json_loads(plan_str)
    logger.info("=" * 100)
    logger.info(f"Generated Plan Prompt:\n{approach_to_plan_prompt}")
    logger.info(f"Generated Plan:\n{steps}")
    logger.info("=" * 100)

    return plan_str, steps


def plan_to_subtasks(
    plans,
    model_name,
    key,
    url,
    model,
    slm,
    sub_retrieval_method,
    top_k,
):
    # ====== [1] Augement plan with required_outcomes ====== #
    generate_required_outcomes_prompt = populate_template(
        planning_prompt_template["generate_required_outcomes_prompt"],
        variables={
            "task": task_text,
            "plan": steps_only_plan,
        },
    )
    # ====== [2] Bundle outcomes ====== #
    required_outcomes_str = call_model(
        query=generate_required_outcomes_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    plan_with_outcomes = _safe_json_loads(required_outcomes_str) or {}
    enriched_plan = plan_with_outcomes.get("plan", [])

    all_outcomes: List[str] = []
    for item in enriched_plan:
        ros = item.get("required_outcomes") or []
        if isinstance(ros, list):
            all_outcomes.extend([str(x) for x in ros])
        else:
            all_outcomes.append(str(ros))

    bundle_outcomes_prompt = populate_template(
        planning_prompt_template["bundle_outcomes_prompt"],
        variables={
            "all_outcomes": all_outcomes,
        },
    )

    bundle_str = call_model(
        query=bundle_outcomes_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    bundle_obj = _safe_json_loads(bundle_str) or {}
    bundles = bundle_obj.get("bundles", [])

    # ====== [3] Outcome bundle to subtask ====== #
    bundle_to_subtask_text_prompt = populate_template(
        planning_prompt_template["bundle_to_subtak_text_prompt"],
        variables={"bundles": bundles},
    )
    subtask_texts_str = call_model(
        query=bundle_to_subtask_text_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    subtask_texts_obj = _safe_json_loads(subtask_texts_str) or {}
    subtask_text_items = subtask_texts_obj.get("subtasks", [])

    new_subtask_records: List[Dict[str, Any]] = []
    for s_idx, st in enumerate(subtask_text_items, start=1):
        bundle_id = st.get("bundle_id", f"RO{s_idx}")
        required_outcomes = st.get("required_outcomes") or []
        subtask_text = st.get("subtask_text", "")

        # outcome_to_subtasks_prompt 는 {} 포맷이라 populate_template가 아니라 format이 더 안전
        outcome_to_subtasks_prompt = planning_prompt_template[
            "outcome_to_subtasks_prompt"
        ].format(
            required_outcomes=required_outcomes,
            subtask_text=subtask_text,
        )

        s2b_str = call_model(
            query=outcome_to_subtasks_prompt,
            model_name=model_name,
            key=key,
            url=url,
            model=model,
            slm=slm,
        )
        s2b = _safe_json_loads(s2b_str) or {}

        record = {
            "subtask_id": f"subtask_{s_idx:03d}",
            "bundle_id": bundle_id,
            "subtask": s2b.get("subtask_title", ""),
            "subtask_text": subtask_text,
            "required_outcomes": required_outcomes,
            "checklist": s2b.get("checklist", []),
            "task_spec": s2b.get("task_spec", {}),
        }
        new_subtask_records.append(record)

    subtask_and_plans = []
    # ====== [4] Gather subtask plans ====== #
    for rec in new_subtask_records:
        subtask_title = rec.get("subtask", "")
        subtask_text = rec.get("subtask_text", "")

        # retrieval query는 title 우선, 없으면 text
        retrieval_query = subtask_title or subtask_text

        retrieval_results = sub_retrieval_method(retrieval_query, top_k=top_k)
        examples = build_similar_subtask_direction_blocks(similars=retrieval_results)

        subtask_plan_prompt = populate_template(
            planning_prompt_template["subtask_plan_prmopt"],
            variables={
                "subtask": subtask_text or subtask_title,
                "examples": examples,
            },
        )

        subtask_plan_str = call_model(
            query=subtask_plan_prompt,
            model_name=model_name,
            key=key,
            url=url,
            model=model,
            slm=slm,
        )
        subtask_plan_obj = _safe_json_loads(subtask_plan_str) or {}
        subtask_plan = subtask_plan_obj.get("plan", [])
        subtask_and_plans.append(
            f"Subtask {subtask_title}: {subtask_text}\nPlan:{subtask_plan}"
        )
    subtask_and_plans = "\n\n".join(subtask_and_plans)
    return subtask_and_plans
