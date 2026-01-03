from __future__ import annotations


from smolagents.agents import populate_template
import yaml
from agent_kb.agent_kb_utils import call_model
import ast
import json
import re
from typing import Any, Dict, List

from .inter_mece import InterMeceEngine, SimInterMeceEngine
from .intra_mece import IntraMeceEngine, SimIntraMeceEngine


def extract_steps(step_str, model_name, key, url, model, slm):
    print("extract steps.")
    extract_steps_prompt_template = load_prompts(
        path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/planner_kb/rationale_planner_prompts.yaml"
    )
    extract_step_number_prompt = populate_template(
        extract_steps_prompt_template["count_steps"],
        variables={"steps": step_str},
    )
    step_number = call_model(
        extract_step_number_prompt, model_name, key, url, model, slm
    )
    step_number = int(step_number)
    print(f"extract {step_number} steps.")

    steps = []
    for i in range(step_number):
        extract_specific_step_prompt = populate_template(
            extract_steps_prompt_template["extract_specific_step"],
            variables={"step_number": step_number + 1, "steps": step_str},
        )
        step = call_model(
            extract_specific_step_prompt, model_name, key, url, model, slm
        )
        steps.appen(step)
    return steps


def parse_steps(output: str) -> List[str]:
    print(f"parse_steps output:{output}")
    """
    Supported patterns (앞뒤에 잡소리 텍스트가 있어도 허용):

    1) JSON 배열:
       ["Step 1: ...", "Step 2: ..."]

    2) 파이썬 리스트/튜플 리터럴:
       ['Step 1: ...', 'Step 2: ...']
       ("Step 1: ...", "Step 2: ...")

    3) 그냥 문자열 나열:
       "Step 1: ...", "Step 2: ...", "Step 3: ..."

    4) Markdown 번호 리스트:
       1. Step 1 ...
          - sub bullet ...

       2. Step 2 ...
          - ...

    5) 'Step 1: ...', 'Step 2: ...' 처럼
       각 Step 헤더와 그 아래 여러 줄이 한 블록인 형식
    """
    text = output.strip()

    # 1) 코드블럭 제거 (``` ... ```)
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 2) 우선 [] / () 로 전체가 감싸져 있는 경우 처리
    stripped = text.lstrip()
    candidate = None
    result = None

    # (a) 텍스트 맨 앞이 [ 또는 ( 인 경우
    if stripped.startswith(("[", "(")):
        candidate = stripped
    else:
        # (b) "output str:(...)" 같은 형태: 콜론/등호 뒤에서부터 [ 또는 ( 찾기
        m = re.search(r"[:=]\s*([\[\(].*)$", text, re.DOTALL)
        if m:
            candidate = m.group(1).strip()

    if candidate:
        try:
            if candidate.startswith("["):
                # 먼저 JSON 시도
                try:
                    result = json.loads(candidate)
                except json.JSONDecodeError:
                    # 안 되면 파이썬 literal 로
                    result = ast.literal_eval(candidate)
            else:
                # "(" 로 시작하면 파이썬 tuple/list literal 로 간주
                result = ast.literal_eval(candidate)

            if isinstance(result, tuple):
                result = list(result)

            if isinstance(result, list) and all(isinstance(x, str) for x in result):
                return result
        except Exception:
            # candidate 파싱 실패하면 fallback 으로
            result = None

    # 3) Fallback 1: Markdown 번호 리스트 (1. ..., 2. ..., ...)
    #    각 번호 블록을 하나의 step 으로 취급
    numbered_pattern = r"^\s*\d+\.\s+(.+?)(?=^\s*\d+\.|\Z)"
    blocks = re.findall(
        numbered_pattern,
        text.strip(),
        flags=re.MULTILINE | re.DOTALL,
    )
    if blocks:
        steps = [b.strip() for b in blocks]
        return steps

    # 3.5) Fallback 1.5: "Step N:" 으로 시작하는 블록들
    #      예: Step 1: ... \n - ... \n\n Step 2: ...
    step_pattern = r"^Step\s+\d+:\s*(.*?)(?=^Step\s+\d+:\s*|\Z)"
    step_blocks = re.findall(
        step_pattern,
        text.strip(),
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if step_blocks:
        steps = [b.strip() for b in step_blocks]
        return steps

    # 4) Fallback 2: 따옴표로 된 문자열들을 전부 추출해서 리스트로 사용
    #   -> "Step 1: ...", "Step 2: ...", ... 같은 케이스 처리
    pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"|\'([^\'\\]*(?:\\.[^\'\\]*)*)\''
    matches = re.findall(pattern, text)

    if not matches:
        raise ValueError(f"Could not parse steps from: {output!r}")

    steps: List[str] = []
    for g1, g2 in matches:
        s = g1 or g2  # "..." 에서 잡힌 그룹 또는 '...' 에서 잡힌 그룹
        steps.append(s)

    if not steps:
        raise ValueError("No steps extracted from output.")

    return steps


def load_prompts(path):
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)
    return prompts


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
    inter_decomp,
    intra_inter_decomp,
    retrieval_method,
    top_k,
    return_as_str=False,
    multiple_decomp=False,
    mode="loss",
):
    if retrieval_method is None:
        print(f"decompose_task - retrieval_method is None")
        task_decomposition_prompt_template = load_prompts(
            path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/planner_kb/rationale_planner_prompts.yaml"
        )["task_decomposition_and_planning_with_icl_examples_prompt"]
        task_decomposition_prompt = populate_template(
            task_decomposition_prompt_template,
            variables={"task": augmented_question},
        )
    else:
        print(f"decompose_task - retrieval_method is not None")
        rationale_retrieval_results = retrieval_method(example["question"], top_k=top_k)
        task_decomposition_prompt_template = load_prompts(
            path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/planner_kb/rationale_planner_prompts.yaml"
        )["task_decomposition_and_planning_with_retrieval_examples_prompt"]
        step_rationale_examples = build_many_entities_examples_no_actions(
            rationale_retrieval_results
        )
        task_decomposition_prompt = populate_template(
            task_decomposition_prompt_template,
            variables={
                "task": augmented_question,
                "retrieval_examples": step_rationale_examples,
            },
        )
    if inter_decomp:
        if mode == "loss":
            engine = InterMeceEngine(
                model,  # (= tm)
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

            best = engine.pick_best(
                task_text=example["question"],
                task_decomposition_prompt=task_decomposition_prompt,
                num_samples=10,
                alpha=0.5,
                min_subtasks=2,
                max_subtasks=10,
                dedup_raw=True,
                seed=None,
                return_topk=1,
            )
        else:
            engine = SimInterMeceEngine(
                tm,
                call_model_fn=call_model_fn,
                call_model_kwargs=call_model_kwargs,
                # embed_texts_fn=embedder,
            )

            best = engine.pick_best(
                task_text=task_text,
                task_decomposition_prompt=prompt,
                num_samples=8,
                alpha=0.5,
                score_kwargs={
                    "coverage_mode": "relu_cos_mean",
                    "redundancy_mode": "abs_cos_mean",
                },
            )

        if best:
            best1 = best[0]
            print("inter_mece:", best1.score)
            print(
                "coverage:", best1.mece.coverage, "exclusivity:", best1.mece.exclusivity
            )
            print("subtasks:", best1.subtasks)
            return "\n".join(best1.subtasks) if return_as_str else best1.subtasks
        else:
            print("No valid decomposition candidates.")
            return ""
    elif intra_inter_decomp:
        if mode == "loss":
            intra_engine = IntraMeceEngine(
                model,  # (= tm)
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

            topk = intra_engine.pick_topk(
                task_text=example["question"],
                task_decomposition_prompt=task_decomposition_prompt,
                num_samples=10,
                top_k=5,
                alpha_inter=0.5,
                min_subtasks=2,
                max_subtasks=10,
            )
        else:
            engine = SimBasedIntraMeceEngine(
                tm,
                call_model_fn=call_model_fn,
                call_model_kwargs=call_model_kwargs,
                embed_texts_fn=my_embedder,  # 있으면 강추
            )
            topk = engine.pick_topk(
                task_text=task_text,
                task_decomposition_prompt=prompt,
                num_samples=20,
                top_k=5,
                alpha_inter=0.5,
                pool_cap=30,
                score_kwargs={
                    "coverage_mode": "relu_cos_mean",
                    "redundancy_mode": "abs_cos_mean",
                },
            )
        if topk:
            best1 = topk[0]
            print("selection_score:", best1.score)
            print("subset_intra_sum:", best1.details["subset_intra_sum"])
            print("inter_mece:", best1.mece.inter_mece)
            print(
                "coverage:", best1.mece.coverage, "exclusivity:", best1.mece.exclusivity
            )
            print("subtasks:", best1.subtasks)
            if multiple_decomp:
                return ["\n".join(cand.subtasks) for cand in topk]
            else:
                return "\n".join(best1.subtasks) if return_as_str else best1.subtasks
        else:
            print("No valid decomposition candidates.")
            return ""
    else:
        task_decomposition_str = call_model(
            query=task_decomposition_prompt,
            model_name=model_name,
            key=key,
            url=url,
            model=model,
            slm=slm,
        )
        print(f"decompose_task - task_decomposition_str: {task_decomposition_str}")
        return (
            task_decomposition_str
            if return_as_str
            else parse_steps(task_decomposition_str)
        )
