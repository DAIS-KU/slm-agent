import torch
from smolagents import TransformersModel


def run_infer(query: str, model) -> str:
    message_content = None
    while message_content is None or message_content.strip() == "":
        messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]
        message = model(messages)
        message_content = message.content
    return message_content


def read_multiline(prompt=">> ", end_token="/end"):
    print(f"여러 줄 입력 모드: 끝내려면 마지막 줄에 {end_token} 입력")
    lines = []
    while True:
        line = input(prompt)
        if line.strip() == end_token:
            break
        lines.append(line)
    return "\n".join(lines)


# https://apidog.com/kr/blog/qwen3-4b-instruct-2507-and-qwen3-4b-thinking-2507-kr/
# https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
def main():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = TransformersModel(
        model_id="/home/huijeong/slm-agent/Qwen3-4B-Instruct-2507",
        device_map="cuda:1" if torch.cuda.is_available() else "cpu",
        torch_dtype=str(dtype).replace("torch.", ""),
        temperature=0.5,
    )

    print("입력 시작! (여러 줄 가능)  |  종료: /q")
    while True:
        first = input(">> ").strip()
        if first == "" or first.lower() == "/q":
            print("종료합니다.")
            break

        rest = read_multiline(prompt=".. ", end_token="/end")
        query = first + ("\n" + rest if rest else "")

        output = run_infer(query, model)
        print("\n[모델 출력]\n" + output)
        print("-" * 60)


if __name__ == "__main__":
    main()
