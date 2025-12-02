import json
from typing import List
import re
import string
import warnings


def normalize_number_str(number_str: str) -> float:
    # we replace these common units and commas to allow
    # conversion to float
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def split_string(
    s: str,
    char_list: list[str] = [",", ";"],
) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def is_float(element: any) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def normalize_str(input_str, remove_punct=True) -> str:
    """
    Normalize a string by:
    - Removing all white spaces
    - Optionally removing punctuation (if remove_punct is True)
    - Converting to lowercase
    Parameters:
    - input_str: str, the string to normalize
    - remove_punct: bool, whether to remove punctuation (default: True)
    Returns:
    - str, the normalized string
    """
    # Remove all white spaces. Required e.g for seagull vs. sea gull
    no_spaces = re.sub(r"\s", "", input_str)

    # Remove punctuation, if specified.
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()


def extract_numbers(text: str) -> List[str]:
    """This pattern matches:
    - Optional negative sign
    - Numbers with optional comma thousand separators
    - Optional decimal points with decimal numbers
    """
    pattern = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

    return [el.replace(",", "") for el in re.findall(pattern, text)]


def get_question_score_gaia(
    model_answer: str,
    ground_truth: str,
) -> bool:
    """Scoring function used to score functions from the GAIA benchmark"""
    if is_float(ground_truth):
        normalized_answer = normalize_number_str(str(model_answer))
        return normalized_answer == float(ground_truth)

    elif any(char in ground_truth for char in [",", ";"]):  # if gt is a list
        # question with the fish: normalization removes punct
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        if len(gt_elems) != len(ma_elems):  # check length is the same
            warnings.warn(
                "Answer lists have different lengths, returning False.", UserWarning
            )
            return False

        comparisons = []
        for ma_elem, gt_elem in zip(
            ma_elems, gt_elems
        ):  # compare each element as float or str
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                # we do not remove punct since comparisons can include punct
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    else:  # if gt is a str
        return normalize_str(model_answer) == normalize_str(ground_truth)


def get_correct(pred, true):
    # pred, true가 모두 숫자인 경우
    if str(true).replace(".", "", 1).isdigit():
        numbers_answer = extract_numbers(str(pred))
        if len(numbers_answer) == 0:
            return False
        return float(numbers_answer[-1]) == float(true)
    else:
        # 숫자가 아닐 경우 다른 스코어 함수 사용
        return get_question_score_gaia(str(pred), str(true))


def evaluate(input_path, output_path):
    total = 0
    correct = 0
    results = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            task_id = data.get("task_id")
            pred = data.get("prediction")
            true = data.get("true_answer")
            is_correct = get_correct(pred, true)
            total += 1
            correct += int(is_correct)

            results.append(
                {
                    "task_id": task_id,
                    "prediction": pred,
                    "true_answer": true,
                    "is_correct": is_correct,
                }
            )

    # 정확도 계산
    accuracy = correct / total if total > 0 else 0

    # 결과 파일로 저장
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 마지막 줄에 점수 정보 추가
        f.write(json.dumps({"accuracy": accuracy}, ensure_ascii=False) + "\n")

    print(f"✅ 총 {total}개 중 {correct}개 정답 ({accuracy:.2%})")
    print(f"결과 파일: {output_path}")


if __name__ == "__main__":
    evaluate(
        input_path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/output/validation/qwen4-base.jsonl",
        output_path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/output/evaluate/qwen4-base.jsonl",
    )
    evaluate(
        input_path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/output/validation/qwen4-subtask.jsonl",
        output_path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/output/evaluate/qwen4-subtask.jsonl",
    )
    evaluate(
        input_path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/output/validation/qwen4-subtask_ex.jsonl",
        output_path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/output/evaluate/qwen4-subtask_ex.jsonl",
    )
    evaluate(
        input_path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/output/validation/qwen4-subtask-prationale.jsonl",
        output_path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/output/evaluate/qwen4-subtask-prationale.jsonl",
    )

    evaluate(
        input_path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/output/validation/qwen4-subtask-prationale_ex.jsonl",
        output_path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/output/evaluate/qwen4-subtask-prationale_ex.jsonl",
    )
    evaluate(
        input_path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/output/validation/qwen4-subtask_ex-prationale_ex.jsonl",
        output_path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/output/evaluate/qwen4-subtask_ex-prationale_ex.jsonl",
    )
    evaluate(
        input_path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/output/validation/qwen4-subtask_ex-prationale_ex.jsonl",
        output_path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/output/evaluate/qwen4-subtask_ex-prationale_ex.jsonl",
    )
