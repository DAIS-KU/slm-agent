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
        elif mode == "task_analysis":
            task_analysis = d.get("task_analysis") or {}
            decision_criterion = task_analysis.get("decision_criterion", "")
            approach = task_analysis.get("approach", "")
            parts.append(f"[Similar Task #{i}] {task}")
            parts.append(f"Decision Criterion: {decision_criterion}")
            parts.append(f"Approach: {approach}\n")
        elif mode == "plan_steps":
            plan_steps_only = d.get("plan_steps_only") or {}
            steps = plan_steps_only.get("steps", [])
            parts.append(f"[Similar Task #{i}] {task}")
            parts.append(f"Steps: {steps}\n")
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


import json
import re
from typing import Any, Dict, Optional

_VALID_JSON_ESCAPES = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}


def _escape_invalid_backslashes(s: str) -> str:
    out = []
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue

        # ch == "\\"
        if i + 1 >= n:
            out.append("\\\\")
            i += 1
            continue

        nxt = s[i + 1]

        if nxt in _VALID_JSON_ESCAPES:
            if nxt == "u":
                # \uXXXX
                if i + 5 < n and all(
                    c in "0123456789abcdefABCDEF" for c in s[i + 2 : i + 6]
                ):
                    out.append("\\u")
                    out.append(s[i + 2 : i + 6])
                    i += 6
                else:
                    out.append("\\\\")
                    i += 1
            else:
                out.append("\\")
                out.append(nxt)
                i += 2
            continue

        # invalid escape: \(
        out.append("\\\\")
        i += 1

    return "".join(out)


def _escape_control_chars_in_strings(s: str) -> str:
    """
    JSON 문자열 내부에 들어간 리터럴 제어문자(개행/탭/CR 등)를 \\n, \\t, \\r 형태로 보정.
    문자열 밖의 개행은 그대로 둬도 무방(공백처럼 처리됨).
    """
    out = []
    in_str = False
    esc = False

    for ch in s:
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            continue

        # in_str == True
        if esc:
            out.append(ch)
            esc = False
            continue

        if ch == "\\":
            out.append(ch)
            esc = True
            continue

        if ch == '"':
            out.append(ch)
            in_str = False
            continue

        # 문자열 내부 제어문자 보정
        if ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)

    return "".join(out)


def _remove_trailing_commas(s: str) -> str:
    # { "a": 1, } / [1,2,] 같은 케이스 보정
    return re.sub(r",(\s*[}\]])", r"\1", s)


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    text에서 첫 번째로 완결되는 JSON object/array 구간만 추출.
    기존의 find('{')~rfind('}')보다 훨씬 안전함(Traceback 등 뒤에 붙어도 OK).
    """
    text = text.strip()
    start = None
    opener = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            opener = ch
            break
    if start is None:
        return None

    stack = []
    in_str = False
    esc = False

    for j in range(start, len(text)):
        ch = text[j]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            top = stack.pop()
            if (top == "{" and ch != "}") or (top == "[" and ch != "]"):
                return None
            if not stack:
                return text[start : j + 1]

    return None


def _safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()

    # 1) 그대로 시도
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Top-level JSON is not an object")
    except Exception:
        # logger가 없다면 print로 대체하거나, 상위에서 logger 주입하세요.
        # logger.info(f"Failed to parse JSON from model output:\n{text}")
        pass

    # 2) 첫 완결 JSON 덩어리만 추출
    candidate = _extract_first_json_object(text)
    if not candidate:
        raise ValueError("No JSON object/array found in text")

    # 3) 단계적 보정 후 재시도 (핵심: 문자열 내부 개행 이스케이프)
    fixed = candidate
    fixed = _escape_invalid_backslashes(fixed)
    fixed = _escape_control_chars_in_strings(fixed)
    fixed = _remove_trailing_commas(fixed)

    try:
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Top-level JSON is not an object")
    except Exception:
        # logger.info(f"Failed to parse JSON after fixes:\n{fixed}")
        raise ValueError("Unable to parse JSON from text")


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


def _build_augmented_plan_reference_blocks(
    similars: List[Any], max_tasks: int = 3, max_steps: int = 5
) -> str:
    """Build reference blocks from augmented_plan for Mode 1 action generation."""
    blocks: List[str] = []
    for i, d in enumerate(similars[:max_tasks], start=1):
        augmented_plan = d.get("augmented_plan") or {}
        steps = augmented_plan.get("steps", [])
        step_parts: List[str] = []
        for step in steps[:max_steps]:
            original = step.get("original_step", "")
            executions = step.get("step_executions", [])
            step_parts.append(
                f"  original_step: {original}\n  step_executions: {executions}"
            )
        if step_parts:
            blocks.append(f"[Reference Task #{i}]\n" + "\n".join(step_parts))
    return "\n\n".join(blocks)


def _build_subtask_actions_blocks(
    subtask_results: List[Any], max_items: int = 3
) -> str:
    """Build action reference blocks from SubtaskKB results for Mode 2."""
    blocks: List[str] = []
    for i, d in enumerate(subtask_results[:max_items], start=1):
        original_step = d.get("original_step")
        task = d.get("task")
        given = d.get("given")
        input_data = d.get("input_data")
        subtask_str = (
            f"Task: {original_step}/n{task}/nGiven: {given}/nInput: {input_data}"
        )
        actions = d.get("actions", [])
        blocks.append(f"[Subtask #{i}] {subtask_str}\nActions: {actions}")
    return "\n\n".join(blocks)


def _format_augmented_plan(steps: List[str], step_actions: List[List[str]]) -> str:
    """Combine plan steps with their generated actions into a readable string."""
    lines: List[str] = []
    for i, (step, actions) in enumerate(zip(steps, step_actions), start=1):
        lines.append(f"Step {i}: {step}")
        if actions:
            for action in actions:
                lines.append(f"  - {action}")
    return "\n".join(lines)


def task_analysis_planning(
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
    mode=None,
    sub_retrieval_method=None,
):
    """Two-step planner grounded in task_analysis fields from the KB.

    Step 1: Retrieve similar tasks and use their task_analysis.decision_criterion /
            task_analysis.approach to generate a question-specific decision_criterion
            and approach.
    Step 2: Use the generated decision_criterion, approach, and the retrieved tasks'
            plan_steps_only.steps to generate question-specific plan steps.

    Optional augmentation modes (applied after Step 2):
        mode=None    – default; return plan_steps_only.steps as-is.
        mode="mode1" – for each step, generate actions using augmented_plan reference
                       blocks from the already-retrieved tasks.
        mode="mode2" – for each step, retrieve matching subtasks from SubtaskKB and
                       generate actions from their actions. Requires sub_retrieval_method.

    Returns:
        plan_str   : str  – formatted plan (steps + actions when mode is set)
        directives : str  – The generated decision_criterion + approach
    """
    prompts = load_prompts(
        path="/home/huijeong/slm-agent/Agent-KB-GAIA/examples/open_deep_research/planner_kb/planner_prompts.yaml"
    )

    retrieval_results = retrieval_method(example["question"], top_k=top_k)

    # ---- Step 1: generate decision_criterion and approach ---- #
    logger.info("=" * 100)
    logger.info(
        "task_analysis_planning – Step 1: generate decision_criterion and approach"
    )

    similar_task_analysis = build_similar_task_blocks(
        retrieval_results, mode="task_analysis"
    )
    task_analysis_prompt = populate_template(
        prompts["task_analysis_prompt"],
        variables={
            "task": augmented_question,
            "similar_blocks": similar_task_analysis,
        },
    )
    task_analysis_str = call_model(
        query=task_analysis_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info(f"task_analysis_prompt:\n{task_analysis_prompt}")
    logger.info(f"task_analysis raw output:\n{task_analysis_str}")

    task_analysis_obj = _safe_json_loads(task_analysis_str) or {}
    decision_criterion = task_analysis_obj.get("decision_criterion", task_analysis_str)
    approach = task_analysis_obj.get("approach", "")
    logger.info(f"decision_criterion: {decision_criterion}")
    logger.info(f"approach: {approach}")
    logger.info("=" * 100)

    # ---- Step 2: generate plan steps ---- #
    logger.info("=" * 100)
    logger.info("task_analysis_planning – Step 2: generate plan steps")

    similar_plan_steps = build_similar_task_blocks(retrieval_results, mode="plan_steps")
    task_analysis_plan_prompt = populate_template(
        prompts["task_analysis_plan_prompt"],
        variables={
            "task": augmented_question,
            "decision_criterion": decision_criterion,
            "approach": approach,
            "similar_blocks": similar_plan_steps,
        },
    )
    plan_str = call_model(
        query=task_analysis_plan_prompt,
        model_name=model_name,
        key=key,
        url=url,
        model=model,
        slm=slm,
    )
    logger.info(f"task_analysis_plan_prompt:\n{task_analysis_plan_prompt}")
    logger.info(f"Generated plan:\n{plan_str}")
    logger.info("=" * 100)

    directives = f"Decision Criterion: {decision_criterion}\nApproach: {approach}"

    if mode is None:
        return plan_str, directives

    # Parse plan steps from generated plan_str
    plan_obj = _safe_json_loads(plan_str) or {}
    plan_steps: List[str] = plan_obj.get("steps", [])
    if not plan_steps:
        logger.warning(
            "task_analysis_planning: could not parse plan steps for augmentation"
        )
        return plan_str, directives

    # ---- Mode 1: augment using augmented_plan reference blocks ---- #
    if mode == "mode1":
        logger.info("=" * 100)
        logger.info("task_analysis_planning – Mode 1: augment with augmented_plan")
        reference_blocks = _build_augmented_plan_reference_blocks(retrieval_results)
        step_actions: List[List[str]] = []
        for step in plan_steps:
            action_prompt = populate_template(
                prompts["step_actions_from_augmented_prompt"],
                variables={"step": step, "reference_blocks": reference_blocks},
            )
            action_str = call_model(
                query=action_prompt,
                model_name=model_name,
                key=key,
                url=url,
                model=model,
                slm=slm,
            )
            action_obj = _safe_json_loads(action_str) or {}
            actions = action_obj.get("actions", [])
            step_actions.append(actions)
            logger.info(f"Step: {step}\nGenerated actions: {actions}")
        logger.info("=" * 100)
        plan_str = _format_augmented_plan(plan_steps, step_actions)
        return plan_str, directives

    # ---- Mode 2: augment using SubtaskKB per-step retrieval ---- #
    if mode == "mode2":
        if sub_retrieval_method is None:
            logger.error(
                "task_analysis_planning mode2: sub_retrieval_method is required"
            )
            return plan_str, directives
        logger.info("=" * 100)
        logger.info("task_analysis_planning – Mode 2: augment with SubtaskKB")
        step_actions = []
        for step in plan_steps:
            sub_results = sub_retrieval_method(step, top_k=top_k)
            retrieved_actions_block = _build_subtask_actions_blocks(sub_results)
            action_prompt = populate_template(
                prompts["step_actions_from_subtask_prompt"],
                variables={"step": step, "retrieved_actions": retrieved_actions_block},
            )
            action_str = call_model(
                query=action_prompt,
                model_name=model_name,
                key=key,
                url=url,
                model=model,
                slm=slm,
            )
            action_obj = _safe_json_loads(action_str) or {}
            actions = action_obj.get("actions", [])
            step_actions.append(actions)
            logger.info(
                f"Step: {step}\nRetrieved subtasks: {[d.get('original_step') for d in sub_results]}\nGenerated actions: {actions}"
            )
        logger.info("=" * 100)
        plan_str = _format_augmented_plan(plan_steps, step_actions)
        return plan_str, directives

    logger.warning(f"task_analysis_planning: unknown mode '{mode}', returning default")
    return plan_str, directives


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
