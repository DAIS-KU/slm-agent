# pip install -U datasets

import json
from datasets import load_dataset

OUT_PATH = "mmlu_dev_one_per_subject.json"


def idx_to_letter(i: int) -> str:
    return "ABCD"[int(i)]


def format_question_with_choices(question: str, choices) -> str:
    # question + "\nA. ...\nB. ...\nC. ...\nD. ..."
    lines = [question.rstrip()]
    for letter, choice in zip("ABCD", choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


# 1) dev split 로드 (config = "all")
# (cais/mmlu는 보통 question / choices(4개) / answer(0~3) / subject 컬럼을 가집니다)
ds = load_dataset(
    "cais/mmlu", "all", split="dev"
)  # dev는 보통 과목당 5문항 정도라 크지 않습니다.

# 2) subject별 1문제씩 선택(앞에서부터 첫 번째)
picked = {}
for ex in ds:
    subj = ex["subject"]
    if subj not in picked:
        picked[subj] = {
            "task_id": subj,
            "question": format_question_with_choices(ex["question"], ex["choices"]),
            "task": format_question_with_choices(ex["question"], ex["choices"]),
            "choices": list(ex["choices"]),
            "answer_index": int(ex["answer"]),
            "true_answer": idx_to_letter(ex["answer"]),
        }

# 3) JSON으로 저장 (subject 기준 정렬)
result = [picked[k] for k in sorted(picked.keys())]

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Saved {len(result)} examples to {OUT_PATH}")
