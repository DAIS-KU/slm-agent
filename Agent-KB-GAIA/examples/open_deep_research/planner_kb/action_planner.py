from smolagents.agents import populate_template
import yaml
from agent_kb.agent_kb_utils import call_model
import ast
import json
import re
from typing import List


def parse_tag_array(text: str) -> List[str]:
    """
    text 안에서 태그 배열 형태의 문자열을 찾아
    파이썬 리스트로 파싱해서 리턴한다.

    지원하는 예시:
      - ["a", "b", "c"]             # JSON 배열
      - ['a', 'b', 'c']             # 파이썬 리스트 literal
      - [a, b, c]                   # 따옴표 없는 배열
      - "a", "b", "c"               # 따옴표 나열
      - a, b, c                     # 쉼표로 구분된 태그 나열
    실패 시에는 예외를 던지지 않고 빈 리스트([])를 반환한다.
    """
    if not isinstance(text, str):
        return []

    text = text.strip()

    # 1) 전체가 그대로 JSON 리스트일 수도 있으니 먼저 시도
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return value
    except Exception:
        pass

    # 2) 내부에서 배열 부분([ ... ])만 뽑기
    start = text.find("[")
    end = text.rfind("]")

    # 2-A) 아예 [] 가 없는 경우
    if start == -1 or end == -1 or end <= start:
        # 2-A-1) 따옴표로 둘러싸인 문자열들을 모두 추출
        candidates = re.findall(r'["\'](.*?)["\']', text)
        candidates = [c.strip() for c in candidates if c.strip()]
        if candidates:
            return candidates

        # 2-A-2) 쉼표로 구분된 문자열을 태그 리스트로 간주
        if "," in text:
            parts = [p.strip() for p in text.split(",")]
            parts = [p for p in parts if p]
            if parts:
                return parts

        # 2-A-3) 쉼표도 따옴표도 없지만 텍스트가 비어있지 않으면
        #        단일 태그로 간주
        if text:
            return [text]

        return []

    # 2-B) [] 가 있는 경우 기존 로직
    array_str = text[start : end + 1]

    # 2-1) JSON 스타일로 다시 시도
    try:
        value = json.loads(array_str)
        if isinstance(value, list):
            return value
    except Exception:
        pass

    # 2-2) 파이썬 literal 형식으로 시도 (단일 따옴표 등)
    try:
        value = ast.literal_eval(array_str)
        if isinstance(value, list):
            return value
    except Exception:
        # literal_eval 실패해도 이제는 에러를 올리지 않고 통과
        pass

    # 2-3) 따옴표 없는 단순 토큰 리스트: [web search, metadata extraction, contextual analysis]
    inner = array_str[1:-1].strip()
    if inner and not re.search(r'["\']', inner):
        parts = [p.strip() for p in inner.split(",")]
        parts = [p for p in parts if p]
        if parts:
            return parts

    # 3) 마지막 fallback: 배열 안의 따옴표 문자열만 추출
    candidates = re.findall(r'["\'](.*?)["\']', array_str)
    candidates = [c.strip() for c in candidates if c.strip()]
    if candidates:
        return candidates

    raise ValueError(f"Cannot parse array from text: {text}")


def load_prompts(path):
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)
    return prompts


def build_action_level_planning_subseq_examples(entities):
    step_and_rationaleses = []
    for entity in entities:
        # print(f"entity :{entity}")
        lines = []
        macro_description = entity.get("macro_description")
        lines.append(f"Similar actions: {macro_description}")
        for i, action in enumerate(entity.get("actions"), start=1):
            action_description = action["action_description"]
            observation = action["observation"]
            lines.append(
                f"{i}. action: {action_description}, observation: {observation}"
            )
        step_and_rationales = "\n".join(lines)
        step_and_rationaleses.append(step_and_rationales)
    step_and_rationaleses_text = "\n".join(step_and_rationaleses)
    return step_and_rationaleses_text


def action_level_planning(
    task, curruent_plans, key, url, model, model_name, retrieval_method, top_k, slm
):
    prompt = load_prompts(
        path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/planner_kb/action_planner_prompts.yaml"
    )
    generate_tag_prompt_template = prompt["generate_tag_prompt"]
    action_planning_results = []
    for plan_number, curruent_plan in enumerate(curruent_plans):
        print(f"action_level_planning #{plan_number}/{len(curruent_plans)}")
        print(f"curruent_plan: {curruent_plan}")
        generate_tag_prompt = populate_template(
            generate_tag_prompt_template,
            variables={"task": task, "curruent_plan": curruent_plan, "top_k": top_k},
        )
        action_tags = call_model(
            query=generate_tag_prompt,
            model_name=model_name,
            key=key,
            url=url,
            model=model,
            slm=slm,
        )
        print(f"action_tags: {action_tags}")
        action_tag_list = parse_tag_array(action_tags)
        print(f"action_tag_list: {action_tag_list}")
        retrieval_results = []
        for action_tag in action_tag_list:
            retrieval_result = retrieval_method(action_tag, top_k=1, is_action=True)
            retrieval_results.extend(retrieval_result)
        action_sequence_examples = build_action_level_planning_subseq_examples(
            retrieval_results
        )

        action_planning_with_examples_prompt_template = prompt[
            "action_planning_with_examples_prompt"
        ]
        action_planning_with_examples_prompt = populate_template(
            action_planning_with_examples_prompt_template,
            variables={
                "task": task,
                "curruent_plan": curruent_plan,
                "action_sequence_examples": action_sequence_examples,
            },
        )
        action_planning_result = call_model(
            query=action_planning_with_examples_prompt,
            model_name=model_name,
            key=key,
            url=url,
            model=model,
            slm=slm,
        )
        action_planning_results.append(action_planning_result)
    return action_planning_results
