import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sentence_transformers import SentenceTransformer

# TODO Action용 분리
@dataclass
class WorkflowInstance:
    """A complete workflow instance"""

    workflow_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    query: str = ""
    agent_planning: Optional[str] = None
    search_agent_planning: Optional[str] = None
    agent_experience: Optional[str] = None
    search_agent_experience: Optional[str] = None
    query_embedding: Optional[np.ndarray] = None
    plan_embedding: Optional[np.ndarray] = None
    search_plan_embedding: Optional[np.ndarray] = None
    steps: Optional[List[str]] = None
    step_rationales: Optional[Dict] = None


@dataclass
class ActionSquenceInstance:
    """A complete action seequence instance"""

    action_sequence_id: str = field(
        default_factory=lambda: str(datetime.now().timestamp())
    )
    macro_type: str
    macro_description: str
    actions: List[Dict]


class AgenticKnowledgeBase:
    def __init__(self, json_file_paths=None):
        self.workflows: Dict[str, WorkflowInstance] = {}
        self.embedding_model = SentenceTransformer(
            "/home/work/huijeong/agent/Agent-KB-GAIA/examples/open_deep_research/models/all-MiniLM-L6-v2",
            # "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.field_components = {
            "query": {
                "vectorizer": TfidfVectorizer(stop_words="english"),
                "matrix": None,
                "workflow_ids": [],
            },
            # 'plan': {
            #     'vectorizer': TfidfVectorizer(stop_words='english'),
            #     'matrix': None,
            #     'workflow_ids': []
            # },
            # 'search_plan': {
            #     'vectorizer': TfidfVectorizer(stop_words='english'),
            #     'matrix': None,
            #     'workflow_ids': []
            # }
        }

        if json_file_paths:
            self.load_initial_data(json_file_paths)
            self.finalize_index()

    def load_initial_data(self, json_file_paths):
        for json_path in json_file_paths:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")
            self.parse_json_file(json_path)

    def parse_json_file(self, json_file_path):
        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)
                batch = []
                for item in data:
                    try:
                        instance = WorkflowInstance(
                            query=item.get("question", ""),
                            agent_planning=item.get("agent_planning"),
                            search_agent_planning=item.get("search_agent_planning"),
                            agent_experience=item.get("agent_experience"),
                            search_agent_experience=item.get("search_agent_experience"),
                            steps=item.get("Steps"),
                            step_rationales=item.get("StepsRationale"),
                        )
                        batch.append(instance)
                    except KeyError as e:
                        print(f"Skipping invalid item: {e}")
                        continue
                for instance in batch:
                    self.workflows[instance.workflow_id] = instance
        except Exception as e:
            print(f"Error parsing file: {e}")

    def add_workflow_instance(self, workflow: WorkflowInstance):
        self.workflows[workflow.workflow_id] = workflow
        return workflow

    def finalize_index(self):
        print("Building search indices...")
        self.build_tfidf_indices()
        self.build_embeddings()

    def build_tfidf_indices(self):
        """Build TF-IDF indices in batch"""
        field_data = {
            "query": [],
            # 'plan': [],
            # 'search_plan': []
        }

        for workflow in self.workflows.values():
            field_data["query"].append(workflow.query)
            # field_data['plan'].append(workflow.agent_planning or "")
            # field_data['search_plan'].append(workflow.search_agent_planning or "")

        # for field in ['query', 'plan', 'search_plan']:
        for field in ["query"]:
            if len(field_data[field]) == 0:
                continue

            vectorizer = self.field_components[field]["vectorizer"]
            self.field_components[field]["matrix"] = vectorizer.fit_transform(
                field_data[field]
            )
            self.field_components[field]["workflow_ids"] = list(self.workflows.keys())

    def build_embeddings(self):
        print("Generating embeddings...")
        workflows = list(self.workflows.values())
        batch_size = 32

        queries = [w.query for w in workflows]
        query_embeddings = self.embedding_model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # plans = [w.agent_planning or "" for w in workflows]
        # plan_embeddings = self.embedding_model.encode(
        #     plans,
        #     batch_size=batch_size,
        #     show_progress_bar=True,
        #     convert_to_numpy=True
        # )

        # search_plans = [w.search_agent_planning or "" for w in workflows]
        # search_plan_embeddings = self.embedding_model.encode(
        #     search_plans,
        #     batch_size=batch_size,
        #     show_progress_bar=True,
        #     convert_to_numpy=True
        # )

        for i, workflow in enumerate(workflows):
            workflow.query_embedding = query_embeddings[i]
            # workflow.plan_embedding = plan_embeddings[i]
            # workflow.search_plan_embedding = search_plan_embeddings[i]

    def field_text_search(self, query: str, field: str, top_k: int = 3) -> List[dict]:
        component = self.field_components[field]
        if component["matrix"] is None or not component["workflow_ids"]:
            return []

        query_vec = component["vectorizer"].transform([query])
        similarities = cosine_similarity(query_vec, component["matrix"]).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [
            {
                "workflow_id": component["workflow_ids"][idx],
                "score": float(similarities[idx]),
                "field": field,
                "content": getattr(
                    self.workflows[component["workflow_ids"][idx]],
                    field if field != "search_plan" else "search_agent_planning",
                ),
            }
            for idx in top_indices
        ]

    def field_semantic_search(
        self, query: str, field: str, top_k: int = 3
    ) -> List[dict]:
        """Optimized semantic search"""
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        embedding_field_map = {
            "query": "query_embedding",
            # 'plan': 'plan_embedding',
            # 'search_plan': 'search_plan_embedding'
        }

        content_field_map = {
            "query": "query",
            # 'plan': 'agent_planning',
            # 'search_plan': 'search_agent_planning'
        }

        embeddings = []
        workflows = []
        for wf_id, workflow in self.workflows.items():
            emb = getattr(workflow, embedding_field_map[field], None)
            if emb is not None:
                embeddings.append(emb)
                workflows.append(workflow)

        if not embeddings:
            return []

        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [
            {
                "workflow_id": workflows[idx].workflow_id,
                "score": float(similarities[idx]),
                "field": field,
                "content": getattr(workflows[idx], content_field_map[field], ""),
            }
            for idx in top_indices
        ]


class ActionAgenticKnowledgeBase:
    def __init__(self, json_file_paths=None):
        self.action_sequences: Dict[str, ActionSquenceInstance] = {}
        self.embedding_model = SentenceTransformer(
            "/home/work/huijeong/agent/Agent-KB-GAIA/examples/open_deep_research/models/all-MiniLM-L6-v2",
            # "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.field_components = {
            "macro_type": {
                "vectorizer": TfidfVectorizer(stop_words="english"),
                "matrix": None,
                "workflow_ids": [],
            },
            # "macro_description": {
            #     "vectorizer": TfidfVectorizer(stop_words="english"),
            #     "matrix": None,
            #     "workflow_ids": [],
            # },
        }

        if json_file_paths:
            self.load_initial_data(json_file_paths)
            self.finalize_index()

    def load_initial_data(self, json_file_paths):
        for json_path in json_file_paths:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")
            self.parse_json_file(json_path)

    def parse_json_file(self, json_file_path):
        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)
                batch = []
                for item in data:
                    try:
                        instance = ActionSquenceInstance(
                            macro_type=item.get("macro_type", ""),
                            macro_description=item.get("macro_description"),
                            actions=item.get("actions"),
                        )
                        batch.append(instance)
                    except KeyError as e:
                        print(f"Skipping invalid item: {e}")
                        continue
                for instance in batch:
                    self.action_sequences[instance.action_sequence_id] = instance
        except Exception as e:
            print(f"Error parsing file: {e}")

    def add_action_sequence_instance(self, action_sequence: ActionSquenceInstance):
        self.action_sequences[action_sequence.action_sequence_id] = action_sequence
        return action_sequence

    def finalize_index(self):
        print("Building search indices...")
        self.build_tfidf_indices()
        self.build_embeddings()

    def build_tfidf_indices(self):
        """Build TF-IDF indices in batch"""
        field_data = {
            "macro_type": [],
            # 'macro_description': [],
        }

        for action_sequence in self.action_sequences.values():
            field_data["macro_type"].append(action_sequence.macro_type)
            # field_data['macro_description'].append(action_sequence.macro_description)

        # for field in ['macro_type', 'macro_description']:
        for field in ["macro_type"]:
            if len(field_data[field]) == 0:
                continue

            vectorizer = self.field_components[field]["vectorizer"]
            self.field_components[field]["matrix"] = vectorizer.fit_transform(
                field_data[field]
            )
            self.field_components[field]["action_sequences_ids"] = list(
                self.action_sequences.keys()
            )

    def build_embeddings(self):
        print("Generating embeddings...")
        action_sequences = list(self.action_sequences.values())
        batch_size = 32

        macro_types = [w.macro_type for w in action_sequences]
        macro_type_embeddings = self.embedding_model.encode(
            macro_types,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # macro_descriptions = [w.macro_description for w in action_sequences]
        # macro_description_embeddings = self.embedding_model.encode(
        #     macro_descriptions,
        #     batch_size=batch_size,
        #     show_progress_bar=True,
        #     convert_to_numpy=True
        # )

        for i, action_sequence in enumerate(action_sequences):
            action_sequence.macro_type_embedding = macro_type_embeddings[i]
            # action_sequence.macro_description_embedding = macro_descriptions[i]

    def field_text_search(self, query: str, field: str, top_k: int = 3) -> List[dict]:
        component = self.field_components[field]
        if component["matrix"] is None or not component["action_sequence_ids"]:
            return []

        query_vec = component["vectorizer"].transform([query])
        similarities = cosine_similarity(query_vec, component["matrix"]).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [
            {
                "action_sequence_id": component["action_sequence_ids"][idx],
                "score": float(similarities[idx]),
                "field": field,
                "content": getattr(
                    self.action_sequences[component["action_sequence_ids"][idx]], field
                ),
            }
            for idx in top_indices
        ]

    def field_semantic_search(
        self, query: str, field: str, top_k: int = 3
    ) -> List[dict]:
        """Optimized semantic search"""
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        embedding_field_map = {
            "macro_type": "macro_type_embedding",
            # 'macro_description': 'macro_description',
        }

        content_field_map = {
            "macro_type": "macro_type",
            # 'macro_description': 'macro_description',
        }

        embeddings = []
        action_sequences = []
        for action_id, action_sequence in self.action_sequences.items():
            emb = getattr(action_sequence, embedding_field_map[field], None)
            if emb is not None:
                embeddings.append(emb)
                action_sequences.append(action_sequence)

        if not embeddings:
            return []

        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [
            {
                "action_sequence_id": action_sequences[idx].action_sequence_id,
                "score": float(similarities[idx]),
                "field": field,
                "content": getattr(action_sequences[idx], content_field_map[field], ""),
            }
            for idx in top_indices
        ]


class AKB_Manager:
    def __init__(self, json_file_paths=None):
        self.knowledge_base = AgenticKnowledgeBase(json_file_paths=json_file_paths)

    def hybrid_search(
        self, query: str, top_k: int = 5, weights: Dict[str, float] = None
    ) -> List[dict]:
        weights = weights or {"text": 0.5, "semantic": 0.5}
        # field_weights = {'query': 0.4, 'plan': 0.3, 'search_plan': 0.3}
        field_weights = {"query": 1.0}

        score_board = defaultdict(float)

        # for field in ['query', 'plan', 'search_plan']:
        for field in ["query"]:
            for result in self.knowledge_base.field_text_search(
                query, field, top_k * 2
            ):
                score_board[result["workflow_id"]] += (
                    weights["text"] * field_weights[field] * result["score"]
                )
            for result in self.knowledge_base.field_semantic_search(
                query, field, top_k * 2
            ):
                score_board[result["workflow_id"]] += (
                    weights["semantic"] * field_weights[field] * result["score"]
                )

        sorted_results = sorted(score_board.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        detailed_results = []
        for wf_id, total_score in sorted_results:
            workflow = self.knowledge_base.workflows[wf_id]
            detailed_results.append(
                {
                    "workflow_id": wf_id,
                    "total_score": total_score,
                    "query": workflow.query,
                    "plan": workflow.agent_planning,
                    "search_plan": workflow.search_agent_planning,
                    "agent_experience": workflow.agent_experience,
                    "search_agent_experience": workflow.search_agent_experience,
                    "steps": workflow.steps,
                    "step_rationales": workflow.step_rationales,
                }
            )

        return detailed_results

    def search_by_text(
        self, query: str, field: str = "query", top_k: int = 3
    ) -> List[dict]:
        """Batch text search"""
        results = []
        for result in self.knowledge_base.field_text_search(query, field, top_k):
            workflow = self.get_workflow_details(result["workflow_id"])
            results.append(
                {
                    "workflow_id": result["workflow_id"],
                    "score": result["score"],
                    "content": {
                        "query": workflow.query,
                        "plan": workflow.agent_planning,
                        "search_plan": workflow.search_agent_planning,
                        "agent_experience": workflow.agent_experience,
                        "search_agent_experience": workflow.search_agent_experience,
                    },
                }
            )
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def search_by_semantic(
        self, query: str, field: str = "query", top_k: int = 3
    ) -> List[dict]:
        """Batch semantic search"""
        results = []
        for result in self.knowledge_base.field_semantic_search(query, field, top_k):
            workflow = self.get_workflow_details(result["workflow_id"])
            results.append(
                {
                    "workflow_id": result["workflow_id"],
                    "score": result["score"],
                    "content": {
                        "query": workflow.query,
                        "plan": workflow.agent_planning,
                        "search_plan": workflow.search_agent_planning,
                        "agent_experience": workflow.agent_experience,
                        "search_agent_experience": workflow.search_agent_experience,
                    },
                }
            )
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def get_workflow_details(self, workflow_id: str) -> Optional[WorkflowInstance]:
        return self.knowledge_base.workflows.get(workflow_id)


class Action_AKB_Manager:
    def __init__(self, json_file_paths=None):
        self.knowledge_base = ActionAgenticKnowledgeBase(
            json_file_paths=json_file_paths
        )

    def hybrid_search(
        self, query: str, top_k: int = 5, weights: Dict[str, float] = None
    ) -> List[dict]:
        weights = weights or {"text": 0.5, "semantic": 0.5}
        # field_weights = {'macro_type': 0.5, 'macro_description': 0.5}
        field_weights = {"macro_type": 1.0}

        score_board = defaultdict(float)

        # for field in ['macro_type', 'macro_description']:
        for field in ["macro_type"]:
            for result in self.knowledge_base.field_text_search(
                query, field, top_k * 2
            ):
                score_board[result["action_sequence_id"]] += (
                    weights["text"] * field_weights[field] * result["score"]
                )
            for result in self.knowledge_base.field_semantic_search(
                query, field, top_k * 2
            ):
                score_board[result["action_sequence_id"]] += (
                    weights["semantic"] * field_weights[field] * result["score"]
                )

        sorted_results = sorted(score_board.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        detailed_results = []
        for action_id, total_score in sorted_results:
            action_sequence = self.knowledge_base.action_sequences[wf_id]
            detailed_results.append(
                {
                    "action_sequence_id": action_id,
                    "total_score": total_score,
                    "macro_type": action_sequence.macro_type,
                    "macro_description": action_sequence.macro_description,
                    "actions": workflow.actions,
                }
            )

        return detailed_results

    def search_by_text(
        self, query: str, field: str = "query", top_k: int = 3
    ) -> List[dict]:
        """Batch text search"""
        results = []
        for result in self.knowledge_base.field_text_search(query, field, top_k):
            action_sequence = self.get_action_sequence_details(
                result["action_sequence_id"]
            )
            results.append(
                {
                    "workflow_id": result["action_sequence_id"],
                    "score": result["score"],
                    "content": {
                        "macro_type": action_sequence.macro_type,
                        "macro_description": action_sequence.macro_description,
                        "actions": workflow.actions,
                    },
                }
            )
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def search_by_semantic(
        self, query: str, field: str = "query", top_k: int = 3
    ) -> List[dict]:
        """Batch semantic search"""
        results = []
        for result in self.knowledge_base.field_semantic_search(query, field, top_k):
            action_sequence = self.get_action_sequence_details(
                result["action_sequence_id"]
            )
            results.append(
                {
                    "workflow_id": result["action_sequence_id"],
                    "score": result["score"],
                    "content": {
                        "macro_type": action_sequence.macro_type,
                        "macro_description": action_sequence.macro_description,
                        "actions": workflow.actions,
                    },
                }
            )
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def get_action_sequence_details(
        self, action_sequence_id: str
    ) -> Optional[ActionSquenceInstance]:
        return self.knowledge_base.action_sequences.get(action_sequence_id)
