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


import logging

logger = logging.getLogger(__name__)


def call_model(query, model_name, key, url, model, slm=False):
    if slm:
        # logger.info(f"call_model query: {query}")
        message_content = None
        while message_content is None or message_content.strip() == "":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                    ],
                }
            ]
            message = model(messages)
            # logger.info(f"call_model raw_response: {message.content}")
            message_content = message.content
        return message_content
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


class AKBClient:
    def __init__(self, base_url="http://localhost:8005"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        weights: Dict[str, float] = None,
    ) -> List[Dict]:
        endpoint = f"{self.base_url}/search/hybrid"
        payload = {
            "query": query,
            "top_k": top_k,
            "weights": weights or {"text": 0.5, "semantic": 0.5},
        }

        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Hybrid search error: {str(e)}")
            return []

    def text_search(self, query: str, top_k: int = 5) -> List[Dict]:
        endpoint = f"{self.base_url}/search/text"
        payload = {"query": query, "top_k": top_k}

        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Text search error: {str(e)}")
            return []

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        endpoint = f"{self.base_url}/search/semantic"
        payload = {"query": query, "top_k": top_k}

        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Semantic search error: {str(e)}")
            return []

    def td_hybrid_search(
        self,
        query: str,
        type_task_path: str,
        domain_path: str,
        top_k: int = 3,
        weights: Dict[str, float] = None,
    ) -> List[Dict]:
        endpoint = f"{self.base_url}/search/td_hybrid"
        payload = {
            "query": query,
            "type_task_path": type_task_path,
            "domain_path": domain_path,
            "top_k": top_k,
            "weights": weights,
        }

        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"TD hybrid search error: {str(e)}")
            return []

    def type_domain_text_search(
        self,
        query: str,
        task_types: List[str],
        domains: List[str],
        top_k: int = 3,
        weights: Dict[str, float] = None,
    ) -> List[Dict]:
        endpoint = f"{self.base_url}/search/type_domain_text"
        payload = {
            "query": query,
            "task_types": task_types,
            "domains": domains,
            "top_k": top_k,
            "weights": weights,
        }

        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Type-domain text search error: {str(e)}")
            return []


class BUAKBClient:
    """Client for agent_kb_service_bu.py (port 8006, bu-taxonomy KB)."""

    def __init__(self, base_url: str = "http://localhost:8006"):
        self.base_url = base_url
        self.session  = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def taxonomy_search(
        self,
        query:            str,
        top_k:            int  = 3,
        tt_path:          Optional[Dict] = None,
        dom_path:         Optional[Dict] = None,
        min_pool_size:    int  = 20,
        hybrid_weights:   Optional[Dict[str, float]] = None,
        include_proposal: bool = True,
    ) -> List[Dict]:
        """taxonomy-filtered hybrid search + dynamic_proposal_planning."""
        payload = {
            "query":            query,
            "top_k":            top_k,
            "tt_path":          tt_path,
            "dom_path":         dom_path,
            "min_pool_size":    min_pool_size,
            "hybrid_weights":   hybrid_weights or {"text": 0.5, "semantic": 0.5},
            "include_proposal": include_proposal,
        }
        try:
            response = self.session.post(f"{self.base_url}/search/taxonomy", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"BU taxonomy search error: {e}")
            return []

    def hybrid_search(
        self,
        query:   str,
        top_k:   int = 5,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        payload = {
            "query":   query,
            "top_k":   top_k,
            "weights": weights or {"text": 0.5, "semantic": 0.5},
        }
        try:
            response = self.session.post(f"{self.base_url}/search/hybrid", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"BU hybrid search error: {e}")
            return []

    def text_search(self, query: str, top_k: int = 5) -> List[Dict]:
        try:
            response = self.session.post(
                f"{self.base_url}/search/text", json={"query": query, "top_k": top_k}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"BU text search error: {e}")
            return []

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        try:
            response = self.session.post(
                f"{self.base_url}/search/semantic", json={"query": query, "top_k": top_k}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"BU semantic search error: {e}")
            return []


class SubAKBClient:
    def __init__(self, base_url="http://localhost:8003"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        weights: Dict[str, float] = None,
    ) -> List[Dict]:
        endpoint = f"{self.base_url}/search/hybrid"
        payload = {
            "query": query,
            "top_k": top_k,
            "weights": weights or {"text": 0.5, "semantic": 0.5},
        }

        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Hybrid search error: {str(e)}")
            return []

    def text_search(self, query: str, top_k: int = 5) -> List[Dict]:
        endpoint = f"{self.base_url}/search/text"
        payload = {"query": query, "top_k": top_k}

        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Text search error: {str(e)}")
            return []

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        endpoint = f"{self.base_url}/search/semantic"
        payload = {"query": query, "top_k": top_k}

        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Semantic search error: {str(e)}")
            return []
