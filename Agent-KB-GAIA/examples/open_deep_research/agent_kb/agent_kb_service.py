from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from agent_kb_retrieval_unified import AKB_Manager
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import time
import os

MAX_CONCURRENT_SEARCHES = int(os.getenv("MAX_CONCURRENT_SEARCHES", 10))
CACHE_TTL = int(os.getenv("CACHE_TTL", 60))

app = FastAPI(title="Optimized Task KB Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 변환 후 엔티티 DB를 로드한다고 가정
# unified_database.json 내부 각 레코드는 task_id/task/subtasks를 포함해야 함
manager = AKB_Manager(json_file_paths=["./augmented_akb.json"])

performance_stats = {
    "total_requests": 0,
    "avg_response_time": 0.0,
    "last_updated": time.time(),
}

response_cache: Dict[str, Dict[str, Any]] = {}


# -----------------------------
# Pydantic models (변환 후 엔티티 기준)
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 1
    weights: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {"text": 0.5, "semantic": 0.5}
    )


class TaskResponse(BaseModel):
    task_id: str
    task: str
    agent_planning: Optional[Any] = None
    # ===== TaskSpec ===== #
    domain: Optional[Any] = None
    skills: Optional[Any] = None
    objective: Optional[Any] = None
    # ===== Context ===== #
    knowledge: Optional[Any] = None
    constraints: Optional[Any] = None
    instructions: Optional[Any] = None
    approach: Optional[Any] = None
    knowledge_summary: Optional[Any] = None
    constraints_summary: Optional[Any] = None
    instructions_summary: Optional[Any] = None
    approach_summary: Optional[Any] = None
    # ===== Plan ===== #
    plan: Optional[Any] = None
    plan_summary: Optional[Any] = None
    total_score: Optional[float] = None  # hybrid일 때만 있을 수도 있어서 Optional


class PerformanceStats(BaseModel):
    total_requests: int
    avg_response_time: float
    cache_hit_rate: float


def update_performance_stats(response_time: float):
    total_time = (
        performance_stats["avg_response_time"] * performance_stats["total_requests"]
    )
    performance_stats["total_requests"] += 1
    performance_stats["avg_response_time"] = (
        total_time + response_time
    ) / performance_stats["total_requests"]
    performance_stats["last_updated"] = time.time()


def _get_cached(cache_key: str):
    cached = response_cache.get(cache_key)
    if not cached:
        return None
    if time.time() - cached["timestamp"] < CACHE_TTL:
        return cached["data"]
    return None


def _set_cached(cache_key: str, data: Any):
    response_cache[cache_key] = {"timestamp": time.time(), "data": data}


def _extract_task_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": str(item["task_id"]),
        "task": item["task"],
        "agent_planning": item["agent_planning"],
        "domain": item["domain"],
        "skills": item["skills"],
        "objective": item["objective"],
        "knowledge": item["knowledge"],
        "constraints": item["constraints"],
        "instructions": item["instructions"],
        "approach": item["approach"],
        "plan": item.get("plan", None),
        "knowledge_summary": item["knowledge_summary"],
        "constraints_summary": item["constraints_summary"],
        "instructions_summary": item["instructions_summary"],
        "approach_summary": item["approach_summary"],
        "plan_summary": item.get("plan_summary", None),
    }


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/search/hybrid", response_model=List[TaskResponse])
async def hybrid_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"hybrid_{request.query}_{request.top_k}_{request.weights}"

    try:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

        # AKB_Manager.hybrid_search가 변환 후 엔티티를 반환한다고 가정
        results = manager.hybrid_search(
            query=request.query, top_k=request.top_k, weights=request.weights
        )

        response_data: List[TaskResponse] = []
        for item in results:
            core = _extract_task_fields(item)
            response_data.append(
                TaskResponse(
                    task_id=str(core["task_id"]),
                    task=core["task"],
                    agent_planning=core["agent_planning"],
                    domain=core["domain"],
                    skills=core["skills"],
                    objective=core["objective"],
                    knowledge=core["knowledge"],
                    constraints=core["constraints"],
                    instructions=core["instructions"],
                    approach=core["approach"],
                    plan=core["plan"],
                    knowledge_summary=core["knowledge_summary"],
                    constraints_summary=core["constraints_summary"],
                    instructions_summary=core["instructions_summary"],
                    approach_summary=core["approach_summary"],
                    plan_summary=core["plan_summary"],
                    total_score=item.get("total_score"),
                )
            )

        _set_cached(cache_key, response_data)
        update_performance_stats(time.time() - start_time)
        return response_data

    except Exception as e:
        print(f"Hybrid search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


@app.post("/search/text", response_model=List[TaskResponse])
async def text_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"text_{request.query}_{request.top_k}"

    try:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

        # 기존 코드의 "query" 필드 대신 "task" 필드를 대상으로 검색
        raw_results = manager.search_by_text(request.query, "task", request.top_k)

        response_data: List[TaskResponse] = []
        for item in raw_results:
            # item: {"score":..., "content": {...}} 형태를 가정
            core = _extract_task_fields(item)
            response_data.append(
                TaskResponse(
                    task_id=str(core["task_id"]),
                    task=core["task"],
                    agent_planning=core["agent_planning"],
                    domain=core["domain"],
                    skills=core["skills"],
                    objective=core["objective"],
                    knowledge=core["knowledge"],
                    constraints=core["constraints"],
                    instructions=core["instructions"],
                    approach=core["approach"],
                    plan=core["plan"],
                    knowledge_summary=core["knowledge_summary"],
                    constraints_summary=core["constraints_summary"],
                    instructions_summary=core["instructions_summary"],
                    approach_summary=core["approach_summary"],
                    plan_summary=core["plan_summary"],
                    total_score=item.get(
                        "score"
                    ),  # text 검색은 score를 total_score로 매핑
                )
            )

        _set_cached(cache_key, response_data)
        update_performance_stats(time.time() - start_time)
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text search failed: {str(e)}")


@app.post("/search/semantic", response_model=List[TaskResponse])
async def semantic_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"semantic_{request.query}_{request.top_k}"

    try:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

        # 기존 코드의 "query" 필드 대신 "task" 필드를 대상으로 검색
        raw_results = manager.search_by_semantic(request.query, "task", request.top_k)

        response_data: List[TaskResponse] = []
        for item in raw_results:
            core = _extract_task_fields(item)
            response_data.append(
                TaskResponse(
                    task_id=str(core["task_id"]),
                    task=core["task"],
                    agent_planning=core["agent_planning"],
                    domain=core["domain"],
                    skills=core["skills"],
                    objective=core["objective"],
                    knowledge=core["knowledge"],
                    constraints=core["constraints"],
                    instructions=core["instructions"],
                    approach=core["approach"],
                    plan=core["plan"],
                    knowledge_summary=core["knowledge_summary"],
                    constraints_summary=core["constraints_summary"],
                    instructions_summary=core["instructions_summary"],
                    approach_summary=core["approach_summary"],
                    plan_summary=core["plan_summary"],
                    total_score=item.get(
                        "score"
                    ),  # semantic 검색도 score를 total_score로 매핑
                )
            )

        _set_cached(cache_key, response_data)
        update_performance_stats(time.time() - start_time)
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@app.get("/performance", response_model=PerformanceStats)
async def get_performance():
    # TTL 내 캐시 엔트리 비율을 “cache hit rate”로 단순 추정
    cache_hit_rate = (
        sum(
            1
            for v in response_cache.values()
            if time.time() - v["timestamp"] < CACHE_TTL
        )
        / len(response_cache)
        if response_cache
        else 0.0
    )

    return {
        "total_requests": performance_stats["total_requests"],
        "avg_response_time": performance_stats["avg_response_time"],
        "cache_hit_rate": cache_hit_rate,
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=int(os.getenv("UVICORN_WORKERS", 1)),
        limit_concurrency=MAX_CONCURRENT_SEARCHES,
    )
