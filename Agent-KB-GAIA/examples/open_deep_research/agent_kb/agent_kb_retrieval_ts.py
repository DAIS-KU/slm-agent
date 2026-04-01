import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class AgenticKnowledgeBase:
    def __init__(self, json_file_paths=None):
        # raw JSON dict 그대로 보존 (task_id -> dict)
        self.tasks: Dict[str, dict] = {}

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.task_ids: List[str] = []
        self.task_embeddings: Optional[np.ndarray] = None

        if json_file_paths:
            self.load_initial_data(json_file_paths)
            self.finalize_index()

    def load_initial_data(self, json_file_paths: List[str]):
        for json_path in json_file_paths:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")
            self.parse_json_file(json_path)

    def parse_json_file(self, json_file_path: str):
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                if json_file_path.endswith(".jsonl"):
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    data = json.load(f)

            if isinstance(data, dict):
                data = [data]

            base_name = os.path.basename(json_file_path)
            for idx, item in enumerate(data):
                try:
                    task_id = item.get("task_id") or f"{base_name}_{idx}"
                    self.tasks[task_id] = item
                except Exception as e:
                    print(f"Skipping invalid item: {e}")
                    continue

        except Exception as e:
            print(f"Error parsing file: {e}")

    def finalize_index(self):
        print("Building search indices...")
        self._build_tfidf_index()
        self._build_embeddings()

    def _build_tfidf_index(self):
        self.task_ids = list(self.tasks.keys())
        texts = [
            self.tasks[tid].get("task") or self.tasks[tid].get("question", "")
            for tid in self.task_ids
        ]
        if not texts:
            return
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

    def _build_embeddings(self):
        print("Generating embeddings...")
        if not self.task_ids:
            return
        texts = [
            self.tasks[tid].get("task") or self.tasks[tid].get("question", "")
            for tid in self.task_ids
        ]
        self.task_embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def field_text_search(self, query: str, top_k: int = 3) -> List[dict]:
        if self.tfidf_matrix is None or not self.task_ids:
            return []
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [
            {"task_id": self.task_ids[idx], "score": float(similarities[idx])}
            for idx in top_indices
        ]

    def field_semantic_search(self, query: str, top_k: int = 3) -> List[dict]:
        if self.task_embeddings is None or not self.task_ids:
            return []
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        similarities = cosine_similarity([query_embedding], self.task_embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [
            {"task_id": self.task_ids[idx], "score": float(similarities[idx])}
            for idx in top_indices
        ]


def _build_result(task_id: str, item: dict, score: float, score_key: str = "total_score") -> dict:
    return {
        "task_id": task_id,
        score_key: score,
        "question": item.get("task") or item.get("question", ""),
        "agent_planning": item.get("agent_planning"),
        "agent_experience": item.get("agent_experience"),
    }


class AKB_Manager:
    def __init__(self, json_file_paths=None):
        self.knowledge_base = AgenticKnowledgeBase(json_file_paths=json_file_paths)

    def hybrid_search(
        self, query: str, top_k: int = 5, weights: Dict[str, float] = None
    ) -> List[dict]:
        weights = weights or {"text": 0.5, "semantic": 0.5}
        cand_k = top_k * 4

        score_board = defaultdict(float)

        for r in self.knowledge_base.field_text_search(query, cand_k):
            score_board[r["task_id"]] += weights["text"] * r["score"]

        for r in self.knowledge_base.field_semantic_search(query, cand_k):
            score_board[r["task_id"]] += weights["semantic"] * r["score"]

        sorted_results = sorted(score_board.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            _build_result(task_id, self.knowledge_base.tasks.get(task_id, {}), total_score)
            for task_id, total_score in sorted_results
        ]

    def search_by_text(self, query: str, field: str = "task", top_k: int = 3) -> List[dict]:
        results = [
            _build_result(r["task_id"], self.knowledge_base.tasks.get(r["task_id"], {}), r["score"], "score")
            for r in self.knowledge_base.field_text_search(query, top_k)
        ]
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def search_by_semantic(self, query: str, field: str = "task", top_k: int = 3) -> List[dict]:
        results = [
            _build_result(r["task_id"], self.knowledge_base.tasks.get(r["task_id"], {}), r["score"], "score")
            for r in self.knowledge_base.field_semantic_search(query, top_k)
        ]
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
