import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# -----------------------------
# Data models (변환 후 엔티티)
# -----------------------------
@dataclass
class Subtask:
    subgoal: str
    rationale: str
    actions: List[str]


@dataclass
class TaskInstance:
    """A converted task entity instance"""

    task_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    task: str = ""
    subtasks: List[Subtask] = field(default_factory=list)

    # Search indices
    task_embedding: Optional[np.ndarray] = None


# -----------------------------
# Knowledge base
# -----------------------------
class AgenticKnowledgeBase:
    def __init__(self, json_file_paths=None):
        self.tasks: Dict[str, TaskInstance] = {}

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # 텍스트 검색은 "task" 한 필드만 인덱싱 (필요하면 actions까지 확장 가능)
        self.field_components = {
            "task": {
                "vectorizer": TfidfVectorizer(stop_words="english"),
                "matrix": None,
                "task_ids": [],
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
        """
        입력 JSON 포맷 가정:
        [
          {
            "task_id": "...",
            "task": "...",
            "subtasks": [
              {"subgoal": "...", "rationale": "...", "actions": ["...", "..."]}
            ]
          },
          ...
        ]
        """
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                # 단일 객체면 list로 감싸기
                data = [data]

            batch: List[TaskInstance] = []
            for item in data:
                try:
                    task_id = item.get("task_id") or str(datetime.now().timestamp())
                    task_text = item.get("task", "")

                    raw_subtasks = item.get("subtasks", [])
                    subtasks: List[Subtask] = []
                    for st in raw_subtasks:
                        if not isinstance(st, dict):
                            continue
                        subtasks.append(
                            Subtask(
                                subgoal=st.get("subgoal", ""),
                                rationale=st.get("rationale", ""),
                                actions=list(st.get("actions", []) or []),
                            )
                        )

                    instance = TaskInstance(
                        task_id=task_id,
                        task=task_text,
                        subtasks=subtasks,
                    )
                    batch.append(instance)
                except Exception as e:
                    print(f"Skipping invalid item: {e}")
                    continue

            for instance in batch:
                self.tasks[instance.task_id] = instance

        except Exception as e:
            print(f"Error parsing file: {e}")

    def add_task_instance(self, task: TaskInstance) -> TaskInstance:
        self.tasks[task.task_id] = task
        return task

    def finalize_index(self):
        print("Building search indices...")
        self.build_tfidf_indices()
        self.build_embeddings()

    # -----------------------------
    # Build indices
    # -----------------------------
    def build_tfidf_indices(self):
        field_data = {"task": []}

        for task in self.tasks.values():
            field_data["task"].append(task.task)

        if len(field_data["task"]) == 0:
            return

        vectorizer = self.field_components["task"]["vectorizer"]
        self.field_components["task"]["matrix"] = vectorizer.fit_transform(
            field_data["task"]
        )
        self.field_components["task"]["task_ids"] = list(self.tasks.keys())

    def build_embeddings(self):
        print("Generating embeddings...")
        tasks = list(self.tasks.values())
        if not tasks:
            return

        batch_size = 32
        texts = [t.task for t in tasks]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        for i, task in enumerate(tasks):
            task.task_embedding = embeddings[i]

    # -----------------------------
    # Search (Text / Semantic)
    # -----------------------------
    def field_text_search(
        self, query: str, field: str = "task", top_k: int = 3
    ) -> List[dict]:
        if field not in self.field_components:
            return []

        component = self.field_components[field]
        if component["matrix"] is None or not component["task_ids"]:
            return []

        query_vec = component["vectorizer"].transform([query])
        similarities = cosine_similarity(query_vec, component["matrix"]).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            task_id = component["task_ids"][idx]
            results.append(
                {
                    "task_id": task_id,
                    "score": float(similarities[idx]),
                    "field": field,
                    "content": getattr(self.tasks[task_id], field),
                }
            )
        return results

    def field_semantic_search(
        self, query: str, field: str = "task", top_k: int = 3
    ) -> List[dict]:
        if field != "task":
            return []

        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        embeddings = []
        tasks = []
        for t in self.tasks.values():
            if t.task_embedding is not None:
                embeddings.append(t.task_embedding)
                tasks.append(t)

        if not embeddings:
            return []

        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "task_id": tasks[idx].task_id,
                    "score": float(similarities[idx]),
                    "field": field,
                    "content": tasks[idx].task,
                }
            )
        return results


# -----------------------------
# Manager (API에서 쓰기 편한 형태로 결과 반환)
# -----------------------------
class AKB_Manager:
    def __init__(self, json_file_paths=None):
        self.knowledge_base = AgenticKnowledgeBase(json_file_paths=json_file_paths)

    def hybrid_search(
        self, query: str, top_k: int = 5, weights: Dict[str, float] = None
    ) -> List[dict]:
        """
        반환 포맷:
        [
          {
            "task_id": "...",
            "total_score": ...,
            "task": "...",
            "subtasks": [...],
          }
        ]
        """
        weights = weights or {"text": 0.5, "semantic": 0.5}
        field_weights = {"task": 1.0}

        score_board = defaultdict(float)

        # text score
        for result in self.knowledge_base.field_text_search(query, "task", top_k * 2):
            score_board[result["task_id"]] += (
                weights["text"] * field_weights["task"] * result["score"]
            )

        # semantic score
        for result in self.knowledge_base.field_semantic_search(
            query, "task", top_k * 2
        ):
            score_board[result["task_id"]] += (
                weights["semantic"] * field_weights["task"] * result["score"]
            )

        sorted_results = sorted(score_board.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        detailed_results = []
        for task_id, total_score in sorted_results:
            task_obj = self.knowledge_base.tasks[task_id]
            detailed_results.append(
                {
                    "task_id": task_id,
                    "total_score": float(total_score),
                    "task": task_obj.task,
                    "subtasks": [
                        {
                            "subgoal": st.subgoal,
                            "rationale": st.rationale,
                            "actions": st.actions,
                        }
                        for st in task_obj.subtasks
                    ],
                }
            )

        return detailed_results

    def search_by_text(
        self, query: str, field: str = "task", top_k: int = 3
    ) -> List[dict]:
        results = []
        for result in self.knowledge_base.field_text_search(query, field, top_k):
            task_obj = self.get_task_details(result["task_id"])
            results.append(
                {
                    "task_id": result["task_id"],
                    "score": result["score"],
                    "content": {
                        "task_id": task_obj.task_id,
                        "task": task_obj.task,
                        "subtasks": [
                            {
                                "subgoal": st.subgoal,
                                "rationale": st.rationale,
                                "actions": st.actions,
                            }
                            for st in task_obj.subtasks
                        ],
                    },
                }
            )
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def search_by_semantic(
        self, query: str, field: str = "task", top_k: int = 3
    ) -> List[dict]:
        results = []
        for result in self.knowledge_base.field_semantic_search(query, field, top_k):
            task_obj = self.get_task_details(result["task_id"])
            results.append(
                {
                    "task_id": result["task_id"],
                    "score": result["score"],
                    "content": {
                        "task_id": task_obj.task_id,
                        "task": task_obj.task,
                        "subtasks": [
                            {
                                "subgoal": st.subgoal,
                                "rationale": st.rationale,
                                "actions": st.actions,
                            }
                            for st in task_obj.subtasks
                        ],
                    },
                }
            )
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def get_task_details(self, task_id: str) -> Optional[TaskInstance]:
        return self.knowledge_base.tasks.get(task_id)
