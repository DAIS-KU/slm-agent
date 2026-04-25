import os
import json
from smolagents.models import (
    OpenAIServerModel,
    FakeToolCallOpenAIServerModel,
)

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_MODEL_ID = "DAIS/Qwen/Qwen3-4B-Instruct-2507-ea0511ad"


def process_selected_tasks_param(tasks_param):

    if not tasks_param:
        return []
    if len(tasks_param) == 1:
        item = tasks_param[0]
        if os.path.isfile(item):
            if item.endswith(".jsonl"):
                with open(item, "r", encoding="utf-8") as f:
                    return [
                        json.loads(line)["id"] for line in f if "id" in json.loads(line)
                    ]
            else:
                with open(item, "r") as f:
                    return [line.strip() for line in f if line.strip()]

        try:
            return [int(item)]
        except ValueError:
            return [item]

    result = []
    for item in tasks_param:
        try:
            result.append(int(item))
        except ValueError:
            result.append(item)

    return result


def prepare_model_kwargs(model_id, args):
    kwargs = {}
    if "o1" in model_id or "o3" in model_id or "o4" in model_id:
        kwargs["reasoning_effort"] = "high"
    if args.temperature:
        kwargs["temperature"] = args.temperature
    if args.top_p:
        kwargs["top_p"] = args.top_p
    return kwargs


def get_api_model(model_id):

    model_wrapper = OpenAIServerModel
    key = os.getenv("OPENAI_API_KEY")
    url = os.getenv("OPENAI_BASE_URL")

    return model_id, key, url, model_wrapper


def get_together_model(model_id=None):
    """Return (model_id, api_key, base_url, model_wrapper) for TogetherAI."""
    mid = model_id or os.getenv("TOGETHER_MODEL_ID", TOGETHER_MODEL_ID)
    key = os.getenv("TOGETHER_API_KEY", "")
    url = os.getenv("TOGETHER_BASE_URL", TOGETHER_BASE_URL)
    return mid, key, url, OpenAIServerModel
