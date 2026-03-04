import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# -----------------------------
# Entity
# -----------------------------
@dataclass
class SubTaskInstance:
    """A converted subtask entity instance"""

    subtask_id: str
    task_id: str
    subtask: str
    do_raw: Optional[Any] = None
    do_sum: Optional[Any] = None
    expected_output: Optional[Any] = None
    actual_output: Optional[Any] = None
    total_score: Optional[float] = None  # 저장용/참고용

    # Search indices
    subtask_embedding: Optional[np.ndarray] = None


# -----------------------------
# KB
# -----------------------------
class AgenticSubKnowledgeBase:
    def __init__(self, json_file_paths: Optional[List[str]] = None):
        # subtask_id -> SubTaskInstance
        self.subtasks: Dict[str, SubTaskInstance] = {}

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # TF-IDF components per field
        self.field_components = {
            "subtask": {
                "vectorizer": TfidfVectorizer(stop_words="english"),
                "matrix": None,
                "subtask_ids": [],
            }
        }

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
                data = json.load(f)

            if isinstance(data, dict):
                data = [data]

            batch: List[SubTaskInstance] = []
            for item in data:
                try:
                    # IDs
                    subtask_id = item.get("subtask_id") or str(
                        datetime.now().timestamp()
                    )
                    task_id = item.get("task_id") or ""

                    # Main text
                    subtask_text = item.get("subtask", "")

                    instance = SubTaskInstance(
                        subtask_id=str(subtask_id),
                        task_id=str(task_id),
                        subtask=subtask_text,
                        do_raw=item.get("do_raw"),
                        do_sum=item.get("do_sum"),
                        expected_output=item.get("expected_output"),
                        actual_output=item.get("actual_output"),
                        total_score=item.get("total_score"),
                    )
                    batch.append(instance)
                except Exception as e:
                    print(f"Skipping invalid item: {e}")
                    continue

            for instance in batch:
                self.subtasks[instance.subtask_id] = instance

        except Exception as e:
            print(f"Error parsing file: {e}")

    def add_subtask_instance(self, subtask: SubTaskInstance) -> SubTaskInstance:
        self.subtasks[subtask.subtask_id] = subtask
        return subtask

    def finalize_index(self):
        print("Building search indices...")
        self.build_tfidf_indices()
        self.build_embeddings()

    # -----------------------------
    # Build indices
    # -----------------------------
    def build_tfidf_indices(self):
        field_data = {"subtask": []}

        for st in self.subtasks.values():
            field_data["subtask"].append(st.subtask)

        if not field_data["subtask"]:
            return

        vectorizer = self.field_components["subtask"]["vectorizer"]
        self.field_components["subtask"]["matrix"] = vectorizer.fit_transform(
            field_data["subtask"]
        )
        self.field_components["subtask"]["subtask_ids"] = list(self.subtasks.keys())

    def build_embeddings(self):
        print("Generating embeddings...")
        subtasks = list(self.subtasks.values())
        if not subtasks:
            return

        batch_size = 32
        texts = [s.subtask for s in subtasks]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        for i, st in enumerate(subtasks):
            st.subtask_embedding = embeddings[i]

    # -----------------------------
    # Search (Text / Semantic)
    # -----------------------------
    def field_text_search(
        self, query: str, field: str = "subtask", top_k: int = 3
    ) -> List[dict]:
        if field not in self.field_components:
            return []

        component = self.field_components[field]
        if component["matrix"] is None or not component["subtask_ids"]:
            return []

        query_vec = component["vectorizer"].transform([query])
        similarities = cosine_similarity(query_vec, component["matrix"]).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            subtask_id = component["subtask_ids"][idx]
            st = self.subtasks[subtask_id]
            results.append(
                {
                    "subtask_id": subtask_id,
                    "score": float(similarities[idx]),
                    "field": field,
                    "content": st.subtask,  # 필드 텍스트(디버그/참고)
                }
            )
        return results

    def field_semantic_search(
        self, query: str, field: str = "subtask", top_k: int = 3
    ) -> List[dict]:
        if field != "subtask":
            return []

        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        embeddings = []
        subtasks = []
        for st in self.subtasks.values():
            if st.subtask_embedding is not None:
                embeddings.append(st.subtask_embedding)
                subtasks.append(st)

        if not embeddings:
            return []

        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "subtask_id": subtasks[idx].subtask_id,
                    "score": float(similarities[idx]),
                    "field": field,
                    "content": subtasks[idx].subtask,
                }
            )
        return results


# -----------------------------
# Manager (API에서 쓰기 편한 형태)
# -----------------------------
class SubAKB_Manager:
    def __init__(self, json_file_paths: Optional[List[str]] = None):
        self.knowledge_base = AgenticSubKnowledgeBase(json_file_paths=json_file_paths)

    def _to_content_dict(self, st: SubTaskInstance) -> Dict[str, Any]:
        return {
            "subtask_id": st.subtask_id,
            "task_id": st.task_id,
            "subtask": st.subtask,
            "do_raw": st.do_raw,
            "do_sum": st.do_sum,
            "expected_output": st.expected_output,
            "actual_output": st.actual_output,
        }

    def hybrid_search(
        self, query: str, top_k: int = 5, weights: Optional[Dict[str, float]] = None
    ) -> List[dict]:
        weights = weights or {"text": 0.5, "semantic": 0.5}
        field_weights = {"subtask": 1.0}

        score_board = defaultdict(float)

        # text
        for r in self.knowledge_base.field_text_search(query, "subtask", top_k * 2):
            score_board[r["subtask_id"]] += (
                weights["text"] * field_weights["subtask"] * r["score"]
            )

        # semantic
        for r in self.knowledge_base.field_semantic_search(query, "subtask", top_k * 2):
            score_board[r["subtask_id"]] += (
                weights["semantic"] * field_weights["subtask"] * r["score"]
            )

        sorted_results = sorted(score_board.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        detailed_results = []
        for subtask_id, total_score in sorted_results:
            st = self.knowledge_base.subtasks[subtask_id]
            detailed_results.append(
                {
                    "subtask_id": subtask_id,
                    "total_score": float(total_score),
                    "content": self._to_content_dict(st),
                }
            )

        return detailed_results

    def search_by_text(
        self, query: str, field: str = "subtask", top_k: int = 3
    ) -> List[dict]:
        results = []
        for r in self.knowledge_base.field_text_search(query, field, top_k):
            st = self.get_subtask_details(r["subtask_id"])
            if not st:
                continue
            results.append(
                {
                    "subtask_id": r["subtask_id"],
                    "score": r["score"],
                    "content": self._to_content_dict(st),
                }
            )
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def search_by_semantic(
        self, query: str, field: str = "subtask", top_k: int = 3
    ) -> List[dict]:
        results = []
        for r in self.knowledge_base.field_semantic_search(query, field, top_k):
            st = self.get_subtask_details(r["subtask_id"])
            if not st:
                continue
            results.append(
                {
                    "subtask_id": r["subtask_id"],
                    "score": r["score"],
                    "content": self._to_content_dict(st),
                }
            )
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def get_subtask_details(self, subtask_id: str) -> Optional[SubTaskInstance]:
        return self.knowledge_base.subtasks.get(subtask_id)
