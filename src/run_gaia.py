# run_gaia_qwen3.py
import os, re, json
from typing import Optional
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from typing import Any, Dict, List, Optional, Tuple

from smolagents import (
    CodeAgent,
    TransformersModel,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
)
from visual_inspector_tool import VisualInspectorTool
from kb import AKB_Manager, WorkflowInstance

# https://github.com/huggingface/smolagents
# https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py
import string
import warnings
import re as _re

def normalize_number_str(number_str: str) -> float:
    # $, %, , 제거 후 float 변환
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        print(f"String {number_str} cannot be normalized to number str.")
        return float("inf")

def split_string(s: str, char_list: list[str] = [",", ";"]) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return _re.split(pattern, s)

def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    """
    문자열 정규화:
    - 모든 공백 제거
    - (기본) 구두점 제거
    - 소문자화
    """
    no_spaces = _re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()

def question_scorer(model_answer: str, ground_truth: str) -> bool:
    # None 처리
    if model_answer is None:
        model_answer = "None"

    # float 판별 헬퍼
    def is_float(element: any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    # (1) GT가 숫자인 경우: 숫자 완전 일치
    if is_float(ground_truth):
        # print(f"Evaluating {model_answer} as a number.")
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    # (2) GT가 리스트(콤마/세미콜론 포함)인 경우
    elif any(char in ground_truth for char in [",", ";"]):
        # print(f"Evaluating {model_answer} as a comma separated list.")
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        if len(gt_elems) != len(ma_elems):
            warnings.warn("Answer lists have different lengths, returning False.", UserWarning)
            return False

        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                # 리스트 내 문자열 비교는 구두점 제거하지 않음(remove_punct=False)
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False) == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # (3) 그 외: 문자열 비교(공백 제거 + (기본) 구두점 제거)
    else:
        # print(f"Evaluating {model_answer} as a string.")
        return normalize_str(model_answer) == normalize_str(ground_truth)

# https://github.com/OPPO-PersonalAI/Agent-KB/blob/master/Agent-KB-GAIA/examples/open_deep_research/run_gaia.py


def build_agent(model_id):
    # ---- GAIA용 시스템 지시: 정답만 출력(불필요한 말 금지) ----
    # GAIA_INSTRUCTIONS = """
    # Make sure to include code with the correct pattern, for instance:
    #         Thoughts: Your thoughts
    #         <code>
    #         # Your python code here
    #         </code>
    # """

    # https://huggingface.co/Qwen/Qwen3-8B
    model = TransformersModel(
        model_id=model_id,  # requires transformers >= 4.51.0
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=1024,
        temperature=0.2,
        trust_remote_code=True,
    )

    # ---- 웹툴 구성: 검색 + 페이지 방문(스크레이프) ----
    tools = [
        DuckDuckGoSearchTool(max_results=5, rate_limit=1.0),
        VisitWebpageTool(max_output_length=40000),
    ]
  
    AUTHORIZED_IMPORTS = [
        "requests",
        "zipfile",
        "os",
        "pandas",
        "numpy",
        "sympy",
        "json",
        "bs4",
        "pubchempy",
        "xml",
        "yahoo_fin",
        "Bio",
        "sklearn",
        "scipy",
        "pydub",
        "io",
        "PIL",
        "chess",
        "PyPDF2",
        "pptx",
        "torch",
        "datetime",
        "fractions",
        "csv",
        "random",
        "re",
        "sys",
        "shutil",
        "re","math","statistics","json","datetime","itertools",
        "pandas","numpy","bs4","lxml","unicodedata","html",
        "cv2",
        "docx"
    ]
    # https://github.com/huggingface/smolagents/blob/1904dddbf30b090d35cdd39e94ab7639781b9a3c/src/smolagents/agents.py#L1478
    agent = CodeAgent(
        tools=tools,
        model=model,
        # instructions=GAIA_INSTRUCTIONS,
        max_steps=10,  # 과도 루프 방지
        add_base_tools=False,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
    )
    return agent

# ---- 간이 정답 정규화(출력 노이즈 제거) ----
def _normalize(s: str):
    s = str(s).strip()
    m = re.search(r"final answer:\s*(.*)$", s, flags=re.I)
    if m:
        s = m.group(1).strip()
    return s

# ---- GAIA 데이터 로드 & 실행 헬퍼 ----
def load_gaia_split(config_name="2023_level1", split="validation"):
    """
    config_name 예시:
      - 2023_level1 / 2023_level2 / 2023_level3 / 2023_all
    """
    ds = load_dataset("gaia-benchmark/GAIA", config_name, split=split)
    return ds

def try_download_attachment(config_name: str, split: str, file_name: str) -> Optional[str]:
    """
    GAIA benchmark 파일을 로컬 캐시에서 우선 읽고,
    없을 경우 Hugging Face Hub에서 다운로드합니다.
    """
    level = (
        "level1" if "level1" in config_name else
        "level2" if "level2" in config_name else
        "level3" if "level3" in config_name else
        ""
    )
    sub = f"2023/{split}"

    # 로컬 캐시 경로 우선 확인
    local_path = f"/home/work/.cache/huggingface/hub/datasets--gaia-benchmark--GAIA/snapshots/897f2dfbb5c952b5c3c1509e648381f9c7b70316/{sub}/{file_name}"
    if os.path.exists(local_path):
        return local_path

    # 로컬에 없으면 Hugging Face에서 다운로드
    try:
        return hf_hub_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            filename=f"{sub}/{file_name}",
        )
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def solve_one(agent, question: str, attachment_path: Optional[str] = None, workflows: List[dict]= None) -> str:
    prompt = question
    if attachment_path:
        prompt += f"\n\nAttached file path (if needed): {attachment_path}"
    if workflows:
        print(f"Retrieve {len(workflows)} workflows")
        plans = "\n---\n".join(item['content']["plan"] for item in workflows)
        prompt += f"\n\nSimilar answer plans: {plans}"
    out = agent.run(prompt)
    return _normalize(out)

def main():
    CONFIG = os.environ.get("GAIA_CONFIG", "2023_level1")
    SPLIT = os.environ.get("GAIA_SPLIT", "validation")
    N = int(os.environ.get("GAIA_LIMIT", "20"))

    ds = load_gaia_split(CONFIG, SPLIT)
    subset = ds.select(range(min(N, len(ds))))

    results = []
    # (NEW) 타입별 통계 집계
    type_counts = {"number": 0, "list": 0, "string": 0}
    type_correct = {"number": 0, "list": 0, "string": 0}

    def _gt_type(gt: str) -> str:
        # scorer.py의 분기와 동일한 기준
        try:
            float(gt)
            return "number"
        except ValueError:
            pass
        if any(ch in gt for ch in [",",";"]):
            return "list"
        return "string"

    correct = 0
    scored = 0

    # model_id ="microsoft/Phi-3-mini-4k-instruct"
    # model_id ="microsoft/Phi-3-small-128k-instruct"
    # model_id="Qwen/Qwen3-8B"
    # model_id="Qwen/Qwen3-4B"
    model_id="Qwen/Qwen3-4B-Instruct-2507"
    model_name = model_id.rsplit("/", 1)[-1]

    agent = build_agent(model_id)
    kb_manager = AKB_Manager(json_file_paths=["/home/work/.default/huijeong/slm-agent/src/knowledge_base.json"])
    for ex in subset:
        q = ex.get("question") or ex.get("Question") or ""
        gold = ex.get("answer") or ex.get("Final answer") or ""
        file_name = ex.get("file_name")
        attach = try_download_attachment(CONFIG, SPLIT, file_name) if file_name else None
        workflows= kb_manager.search_by_semantic(query=q)

        print(f"Raw ex: {ex}")
        pred = solve_one(agent, q, attach, workflows)
        # pred = solve_one(agent, q, attach)

        ok = None
        if gold:
            ok = question_scorer(pred, gold)  # (NEW) 공개 채점 사용
            scored += 1
            if ok:
                correct += 1
            # 타입 집계
            t = _gt_type(gold)
            type_counts[t] += 1
            if ok:
                type_correct[t] += 1

        results.append({
            "task_id": ex.get("id") or ex.get("task_id"),
            "question": q,
            "model_answer": pred,     # 제출용 필드명
            "gold_answer": gold,      # 로컬 검증 참고
            "file_name": file_name,
            "attachment_local_path": attach,
            "correct": ok
        })

        print(f"[{len(results)}/{len(subset)}] pred={pred}  gold={gold}  ok={ok}")

    # (NEW) 공개 채점 기반 성능 요약
    if scored > 0:
        acc = correct / scored
        print(f"\nGAIA public-scoring accuracy: {acc:.3f}  (correct {correct} / {scored})")

        # 타입별 리포트
        for t in ["number","list","string"]:
            if type_counts[t] > 0:
                ta = type_correct[t] / type_counts[t]
                print(f" - {t}: {ta:.3f}  ({type_correct[t]} / {type_counts[t]})")

    # 제출 파일(jsonl): task_id, model_answer 필수
    sub_path = f"../result/{model_name}_gaia_submission.jsonl"
    with open(sub_path, "w", encoding="utf-8") as f:
        for r in results:
            line = {"task_id": r["task_id"], "model_answer": r["model_answer"]}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"\nWrote submission file: {sub_path}")

    # (선택) 로컬 평가 결과도 저장
    report_path = f"../result/{model_name}_gaia_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"GAIA_CONFIG": CONFIG, "GAIA_SPLIT": SPLIT, "N": len(subset)},
            "overall": {"scored": scored, "correct": correct, "accuracy": (correct / scored) if scored else None},
            "by_type": {
                t: {"count": type_counts[t], "correct": type_correct[t],
                    "accuracy": (type_correct[t]/type_counts[t]) if type_counts[t] else None}
                for t in type_counts
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"Wrote eval report: {report_path}")

if __name__ == "__main__":
    main()
