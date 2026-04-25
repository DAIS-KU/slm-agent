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
# Data models
# -----------------------------
@dataclass
class OriginalContext:
    agent_planning: Optional[Any] = None
    agent_experience: Optional[Any] = None


@dataclass
class AugmentedContext:
    task_analysis: Optional[Any] = None
    plan_subtask_action: Optional[Any] = None
    instance: Optional[str] = None        # decision_augmentation.final_reference.instance
    decision_guide: Optional[Any] = None  # decision_augmentation.signals_summary


@dataclass
class TaskInstance:
    """A converted task entity instance"""

    task_id: str
    task: str
    true_answer: str
    original: OriginalContext
    augmented: AugmentedContext

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
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                if json_file_path.endswith(".jsonl"):
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    data = json.load(f)

            if isinstance(data, dict):
                # 단일 객체면 list로 감싸기
                data = [data]

            batch: List[TaskInstance] = []
            base_name = os.path.basename(json_file_path)
            for idx, item in enumerate(data):
                try:
                    task_id = item.get("task_id") or f"{base_name}_{idx}"
                    # unified_database_akb900 등 "question" 키를 사용하는 포맷 지원
                    task_text = item.get("task") or item.get("question", "")

                    true_answer = item.get("true_answer", "")
                    # unified_database_ab900: agent_planning 없이 conversations만 있는 경우
                    agent_planning = item.get("agent_planning") or item.get("search_agent_planning", None)
                    agent_experience = item.get("agent_experience") or item.get("search_agent_experience", None)

                    task_analysis = item.get("task_analysis", None)
                    plan_subtask_action = item.get("plan_subtask_action", None)
                    da = item.get("decision_augmentation") or {}
                    fr = da.get("final_reference") or {}
                    instance_text = fr.get("instance") or None
                    decision_guide = da.get("signals_summary") or None

                    # Normalize odyseuss_db.jsonl format
                    if task_analysis is not None:
                        ta_type = task_analysis.get("task_type")
                        ta_domain = task_analysis.get("domain")
                        # odyseuss_db: {"raw": "...", "normalized": "..."} → list
                        if isinstance(ta_type, dict):
                            task_analysis["task_type"] = [
                                v for v in [ta_type.get("raw"), ta_type.get("normalized")] if v
                            ]
                            task_analysis["task_type_normalized"] = (
                                [ta_type["normalized"]] if ta_type.get("normalized") else []
                            )
                        else:
                            task_analysis["task_type_normalized"] = list(ta_type or [])
                        if isinstance(ta_domain, dict):
                            task_analysis["domain"] = [
                                v for v in [ta_domain.get("raw"), ta_domain.get("normalized")] if v
                            ]
                            task_analysis["domain_normalized"] = (
                                [ta_domain["normalized"]] if ta_domain.get("normalized") else []
                            )
                        else:
                            task_analysis["domain_normalized"] = list(ta_domain or [])
                        # Inject knowledge/plan/instance from final_reference if missing
                        if not task_analysis.get("knowledge") and fr.get("knowledge"):
                            task_analysis["knowledge"] = fr["knowledge"]
                        if not task_analysis.get("plan") and fr.get("plan"):
                            task_analysis["plan"] = fr["plan"]
                        elif not task_analysis.get("plan") and agent_planning:
                            task_analysis["plan"] = [
                                ln.strip().lstrip("- ").lstrip("* ")
                                for ln in agent_planning.split("\n")
                                if ln.strip()
                            ]

                    instance = TaskInstance(
                        task_id=task_id,
                        task=task_text,
                        true_answer=true_answer,
                        original=OriginalContext(
                            agent_planning=agent_planning,
                            agent_experience=agent_experience,
                        ),
                        augmented=AugmentedContext(
                            task_analysis=task_analysis,
                            plan_subtask_action=plan_subtask_action,
                            instance=instance_text,
                            decision_guide=decision_guide,
                        ),
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
        self.build_bucket_index()

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

    def build_bucket_index(self):
        """Inverted index: normalized_task_type → normalized_domain → [task_ids].
        Used for Stage-1 candidate retrieval before text ranking.
        """
        self.bucket_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for task_id, task_obj in self.tasks.items():
            ta = task_obj.augmented.task_analysis or {}
            for tt in (ta.get("task_type_normalized") or []):
                for dom in (ta.get("domain_normalized") or []):
                    self.bucket_index[tt][dom].append(task_id)
        total = sum(
            len(ids)
            for dom_map in self.bucket_index.values()
            for ids in dom_map.values()
        )
        print(f"Bucket index built: {len(self.bucket_index)} task_types, {total} (type,domain) entries")

    def bucket_text_search(
        self,
        query: str,
        task_types: List[str],
        domains: List[str],
        top_k: int = 3,
    ) -> List[dict]:
        """Two-stage retrieval:
        Stage 1 — collect candidates from normalized (task_type, domain) buckets.
                   Every matching bucket contributes all its documents.
        Stage 2 — rank candidates by TF-IDF cosine similarity, return top_k.
        """
        # Stage 1: bucket lookup
        candidate_ids: List[str] = []
        seen: set = set()
        for tt in task_types:
            for dom in domains:
                for tid in self.bucket_index.get(tt, {}).get(dom, []):
                    if tid not in seen:
                        seen.add(tid)
                        candidate_ids.append(tid)

        if not candidate_ids:
            return []

        # Stage 2: TF-IDF text ranking within candidates
        component = self.field_components["task"]
        if component["matrix"] is None or not component["task_ids"]:
            # No TF-IDF index — return candidates in insertion order up to top_k
            return [
                {"task_id": tid, "score": 0.0, "field": "task"}
                for tid in candidate_ids[:top_k]
            ]

        id_to_pos = {tid: pos for pos, tid in enumerate(component["task_ids"])}
        query_vec = component["vectorizer"].transform([query])

        scored: List[tuple] = []
        for tid in candidate_ids:
            pos = id_to_pos.get(tid)
            if pos is not None:
                score = float((query_vec * component["matrix"][pos].T).toarray()[0][0])
                scored.append((score, tid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"task_id": tid, "score": score, "field": "task"}
            for score, tid in scored[:top_k]
        ]

    # -----------------------------
    # Search (Text / Semantic)
    # -----------------------------
    def type_domain_filtered_text_search(
        self,
        query: str,
        task_types: List[str],
        domains: List[str],
        top_k: int = 3,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[dict]:
        """Score all KB tasks by a weighted sum of three normalized components:
          - type_score  = |query_types & kb_types|  / max(|query_types|,  1)  [0, 1]
          - domain_score = |query_domains & kb_domains| / max(|query_domains|, 1) [0, 1]
          - text_score   = TF-IDF cosine similarity against the query           [0, 1]

        final_score = w_type * type_score + w_domain * domain_score + w_text * text_score

        Returns top_k tasks sorted by final_score DESC.
        KB records store task_type under task_analysis.task_type and domain
        under task_analysis.domain.
        """
        if weights is None:
            weights = {"type": 0.3, "domain": 0.3, "text": 0.4}
        w_type = weights.get("type", 0.3)
        w_domain = weights.get("domain", 0.3)
        w_text = weights.get("text", 0.4)

        type_set = set(task_types)
        domain_set = set(domains)
        n_types = max(len(type_set), 1)
        n_domains = max(len(domain_set), 1)

        # Prepare TF-IDF query vector if available
        component = self.field_components["task"]
        tfidf_available = component["matrix"] is not None and bool(component["task_ids"])
        if tfidf_available:
            query_vec = component["vectorizer"].transform([query])
            id_to_pos = {tid: pos for pos, tid in enumerate(component["task_ids"])}

        scored: List[tuple] = []  # (final_score, type_overlap, domain_overlap, text_score, task_id)
        for task_id, task_obj in self.tasks.items():
            ta = task_obj.augmented.task_analysis or {}
            kb_types = set(ta.get("task_type") or [])
            kb_domains = set(ta.get("domain") or [])

            type_overlap = len(type_set & kb_types)
            domain_overlap = len(domain_set & kb_domains)
            type_score = type_overlap / n_types
            domain_score = domain_overlap / n_domains

            if tfidf_available:
                pos = id_to_pos.get(task_id)
                if pos is not None:
                    row = component["matrix"][pos]
                    text_score = float((query_vec * row.T).toarray()[0][0])
                else:
                    text_score = 0.0
            else:
                text_score = 0.0

            final_score = w_type * type_score + w_domain * domain_score + w_text * text_score
            scored.append((final_score, type_overlap, domain_overlap, text_score, task_id))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for final_score, type_overlap, domain_overlap, text_score, task_id in scored[:top_k]:
            results.append({
                "task_id": task_id,
                "score": final_score,
                "overlap_count": type_overlap + domain_overlap,
                "field": "task",
                "content": self.tasks[task_id].task,
            })
        return results

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
                    "true_answer": task_obj.true_answer,
                    "agent_planning": task_obj.original.agent_planning,
                    "agent_experience": task_obj.original.agent_experience,
                    "task_analysis": task_obj.augmented.task_analysis,
                    "plan_subtask_action": task_obj.augmented.plan_subtask_action,
                    "instance": task_obj.augmented.instance,
                    "decision_guide": task_obj.augmented.decision_guide,
                }
            )

        return detailed_results

    def type_domain_text_search(
        self,
        query: str,
        task_types: List[str],
        domains: List[str],
        top_k: int = 3,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[dict]:
        results = []
        for result in self.knowledge_base.type_domain_filtered_text_search(
            query, task_types, domains, top_k, weights=weights
        ):
            task_obj = self.get_task_details(result["task_id"])
            results.append(
                {
                    "task_id": result["task_id"],
                    "score": result["score"],
                    "overlap_count": result.get("overlap_count", 0),
                    "task": task_obj.task,
                    "true_answer": task_obj.true_answer,
                    "agent_planning": task_obj.original.agent_planning,
                    "agent_experience": task_obj.original.agent_experience,
                    "task_analysis": task_obj.augmented.task_analysis,
                    "plan_subtask_action": task_obj.augmented.plan_subtask_action,
                    "instance": task_obj.augmented.instance,
                    "decision_guide": task_obj.augmented.decision_guide,
                }
            )
        return results

    def _task_to_dict(self, task_obj, score: float) -> dict:
        return {
            "task_id": task_obj.task_id,
            "task": task_obj.task,
            "true_answer": task_obj.true_answer,
            "agent_planning": task_obj.original.agent_planning,
            "agent_experience": task_obj.original.agent_experience,
            "task_analysis": task_obj.augmented.task_analysis,
            "plan_subtask_action": task_obj.augmented.plan_subtask_action,
            "instance": task_obj.augmented.instance,
            "decision_guide": task_obj.augmented.decision_guide,
            "score": score,
        }

    def search_by_text(
        self, query: str, field: str = "task", top_k: int = 3
    ) -> List[dict]:
        results = []
        for result in self.knowledge_base.field_text_search(query, field, top_k):
            task_obj = self.get_task_details(result["task_id"])
            entry = self._task_to_dict(task_obj, result["score"])
            results.append({"task_id": result["task_id"], "score": result["score"], "content": entry})
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def search_by_semantic(
        self, query: str, field: str = "task", top_k: int = 3
    ) -> List[dict]:
        results = []
        for result in self.knowledge_base.field_semantic_search(query, field, top_k):
            task_obj = self.get_task_details(result["task_id"])
            entry = self._task_to_dict(task_obj, result["score"])
            results.append({"task_id": result["task_id"], "score": result["score"], "content": entry})
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def bucket_rank_search(
        self,
        query: str,
        task_types: List[str],
        domains: List[str],
        top_k: int = 3,
    ) -> List[dict]:
        """Two-stage retrieval:
        Stage 1 — collect candidates from normalized (task_type × domain) buckets.
        Stage 2 — rank by TF-IDF text similarity, return top_k.
        Falls back to hybrid_search when no bucket candidates found.
        """
        results = self.knowledge_base.bucket_text_search(query, task_types, domains, top_k)
        if not results:
            return self.hybrid_search(query, top_k=top_k)
        detailed = []
        for r in results:
            task_obj = self.get_task_details(r["task_id"])
            detailed.append(self._task_to_dict(task_obj, r["score"]))
        return detailed

    def get_task_details(self, task_id: str) -> Optional[TaskInstance]:
        return self.knowledge_base.tasks.get(task_id)
