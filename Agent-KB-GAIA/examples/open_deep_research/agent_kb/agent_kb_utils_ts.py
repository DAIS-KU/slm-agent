from typing import Dict, List

import requests


class AKBClientTS:
    """HTTP client for agent_kb_service_ts (unified_database KB)."""

    def __init__(self, base_url="http://localhost:8006"):
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
            print(f"TS hybrid search error: {str(e)}")
            return []

    def text_search(self, query: str, top_k: int = 5) -> List[Dict]:
        endpoint = f"{self.base_url}/search/text"
        payload = {"query": query, "top_k": top_k}
        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"TS text search error: {str(e)}")
            return []

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        endpoint = f"{self.base_url}/search/semantic"
        payload = {"query": query, "top_k": top_k}
        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"TS semantic search error: {str(e)}")
            return []


def build_additional_knowledge(results: List[Dict]) -> str:
    """Format ts KB search results into an additional_knowledge prefix string."""
    if not results:
        return ""
    blocks = []
    for i, item in enumerate(results, 1):
        task = item.get("task", "")
        agent_planning = item.get("agent_planning") or ""
        block = f"[Similar Task #{i}] {task}"
        if agent_planning:
            block += f"\nAgent Planning: {agent_planning}"
        blocks.append(block)
    return "Here are similar task examples for reference:\n\n" + "\n\n".join(blocks)
