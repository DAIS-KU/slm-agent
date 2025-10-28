import json

def evaluate(input_path, output_path):
    total = 0
    correct = 0
    results = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            pred = data.get("prediction")
            true = data.get("true_answer")
            is_correct = pred == true
            total += 1
            correct += int(is_correct)

            results.append({
                "prediction": pred,
                "true_answer": true,
                "is_correct": is_correct
            })

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
    evaluate(input_path="", output_path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/output/evaluate")