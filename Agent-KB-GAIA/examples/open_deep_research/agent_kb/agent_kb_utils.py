from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, TypedDict, Union
from openai import OpenAI

import requests

def call_model(query, model_name, key, url, model, slm=False):
    if len(query) > 300000:
        query = query[:300000]
    if slm:
        messages = [{
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": query
                },
            ]
            }]
        message = model(messages)
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
                    {
                    "type": "text",
                    "text": query
                    },
                ]
                }
            ]
        )
        return completion.choices[0].message.content



class AKBClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def hybrid_search(self, query: str, top_k: int = 5, weights: Dict[str, float] = None) -> List[Dict]:
        endpoint = f"{self.base_url}/search/hybrid"
        payload = {
            "query": query,
            "top_k": top_k,
            "weights": weights or {"text": 0.5, "semantic": 0.5}
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