import requests
from typing import Dict, List, Optional, TypedDict
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    Union,
)
from openai import OpenAI

import requests


def call_model(query, model_name, key, url, model, slm=False):
    if len(query) > 300000:
        query = query[:300000]
    if slm:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                ],
            }
        ]
        message = model(messages)
        # print(f"call_model raw_response: {message}")
        return message.content
    else:
        client = OpenAI(
            base_url=url,
            api_key=key,
        )
        completion = client.chat.completions.create(
            extra_body={},
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                    ],
                }
            ],
        )
        return completion.choices[0].message.content


class Subtask(TypedDict):
    subgoal: str
    rationale: str
    actions: List[str]


class TaskResult(TypedDict, total=False):
    task_id: str
    task: str
    subtasks: List[Subtask]
    total_score: float  # hybrid일 때 포함될 수 있음


class AKBClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[TaskResult]:
        endpoint = f"{self.base_url}/search/hybrid"
        payload = {
            "query": query,
            "top_k": top_k,
            "weights": weights or {"text": 0.5, "semantic": 0.5},
        }

        try:
            response = self.session.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Hybrid search error: {str(e)}")
            return []

    def text_search(self, query: str, top_k: int = 5) -> List[TaskResult]:
        endpoint = f"{self.base_url}/search/text"
        payload = {"query": query, "top_k": top_k}

        try:
            response = self.session.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Text search error: {str(e)}")
            return []

    def semantic_search(self, query: str, top_k: int = 5) -> List[TaskResult]:
        endpoint = f"{self.base_url}/search/semantic"
        payload = {"query": query, "top_k": top_k}

        try:
            response = self.session.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Semantic search error: {str(e)}")
            return []
