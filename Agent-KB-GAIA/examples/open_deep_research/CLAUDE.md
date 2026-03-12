# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Start the Agent Knowledge Base (AKB) service
The retrieval service must be running before `run_cr.py` can use KB retrieval:
```bash
# Main task KB service (port 8000)
cd agent_kb && python agent_kb_service.py

# Subtask KB service (port 8001, optional)
cd agent_kb && python subtask_kb_service.py
```

### Run the main agent evaluation
```bash
python run_cr.py \
  --run-name <run_name> \
  --model-id <model_id> \
  --agent_kb \
  --retrieval_type hybrid \
  --top_k 3 \
  --is_progressive
```

Key flags:
- `--agent_kb`: enables AKB retrieval (requires service running)
- `--is_progressive`: uses progressive planning (knowledge -> CI -> approach -> plan)
- `--is_augmented`: uses augmented plan from KB examples
- `--init_plan`: use KB-retrieved plan directly as initial agent plan
- `--slm`: use a local Transformers model instead of OpenAI API
- `--concurrency N`: run N tasks in parallel with ThreadPoolExecutor

### Evaluate predictions
```bash
python evaluate.py
# Edit the hardcoded paths at bottom of evaluate.py before running
```

## Environment Variables

Requires a `.env` file two directories above the project root (or `OPENAI_API_KEY` / `OPENAI_BASE_URL` set in environment):
```
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
HF_TOKEN=...
```

## Architecture

### High-level flow

```
run_cr.py
  -> planner_kb  (generates additional_knowledge from KB)
  -> smolagents CodeAgent  (runs ReAct loop with additional_knowledge)
  -> scripts/reformulator  (cleans up final answer)
  -> evaluate.py  (scores against ground truth)
```

### Agent Knowledge Base (agent_kb/)

Two-layer retrieval system for similar past tasks:

- **`agent_kb_retrieval.py`** — `AgenticKnowledgeBase` builds TF-IDF + sentence-transformer (all-MiniLM-L6-v2) indices over `augmented_akb.json`. `AKB_Manager` exposes `hybrid_search`, `search_by_text`, `search_by_semantic`.
- **`agent_kb_service.py`** — FastAPI wrapper around `AKB_Manager`, serving on port 8000. Endpoints: `POST /search/hybrid`, `POST /search/text`, `POST /search/semantic`.
- **`agent_kb_utils.py`** — `AKBClient` (HTTP client to port 8000) and `SubAKBClient` (port 8001). `call_model()` abstracts SLM vs. OpenAI API calls.
- **`augmented_akb.json`** / **`augmented_ab.json`** — knowledge base records with fields: `task`, `domain`, `skills`, `objective`, `knowledge`, `constraints`, `instructions`, `approach`, `plan` (and `_summary` variants).

### Planner KB (planner_kb/)

Takes a query, retrieves similar tasks from AKB, and synthesizes `additional_knowledge` to prepend to the agent's context:

- **`planner.py`** — main entry points: `planning_task()` (single-pass: plan + CI), `progressive_planning_task()` (staged: knowledge -> CI -> approach -> plan), `recontextulaized_planning_task()` (transfer-based).
- **`planner_prompts.yaml`** — all prompt templates used by the planner.
- **`action_planner.py`** / **`rationale_planner.py`** — alternative planner variants.
- **`inter_mece.py`**, **`intra_mece.py`**, **`incre_mece.py`**, **`mece_common.py`** — MECE-based plan decomposition utilities.

### Scripts (scripts/)

Supporting tools used inside the agent loop:
- **`automodel.py`** — `get_api_model()` resolves model ID to API key/URL/wrapper class; `prepare_model_kwargs()` handles reasoning model params.
- **`run_agents.py`** — `get_single_file_description()` / `get_zip_description()` for inspecting attached files.
- **`reformulator.py`** — `prepare_response()` extracts the final concise answer from agent memory.
- **`searcher.py`** — `SearchTool` wrapping web search (Tavily/Exa/etc.).
- **`async_web_crawler.py`** — `CrawlerReadTool`, `CrawlerArchiveSearchTool`, `SimpleCrawler`.
- **`scorer.py`** / **`gaia_scorer.py`** — GAIA benchmark scoring logic.
- **`text_inspector_tool.py`**, **`audio_inspector_tool.py`**, **`visual_inspector_tool.py`** — multimodal file inspection tools passed to the agent.

### Reflectors (reflectors/)

- **`search_reflector.py`** — `SearchReflector` wraps an LLM to rewrite/expand search queries and reflect on search results. Prompts loaded from `search_prompts.yaml`.

### RAG (rag/)

General-purpose RAG library vendored into the project. Provides:
- Embedding backends: OpenAI, Mistral, Jina, sentence-transformers, VLM
- Vector DB storages: Milvus, Qdrant
- Graph storages: Neo4j, NebulaGraph
- Retrievers: BM25, vector, auto-retriever, Cohere rerank
- Loaders: Firecrawl, Jina URL reader, Chunkr, Apify, Unstructured.io

The RAG library is not directly used by the main agent pipeline; the agent_kb uses its own TF-IDF + sentence-transformers stack directly.

## Data Flow

Tasks are loaded from `kb_tasks.json` (JSON array or JSONL). Already-completed tasks are tracked in the output JSONL file (`output/cr/<run_name>.jsonl`) to allow resuming interrupted runs. Results are evaluated with `evaluate.py` which writes scored output to `output/evaluate/`.

## Key Hardcoded Paths

Several files contain absolute paths to `/home/huijeong/slm-agent/...` that must be updated when running in a different environment:
- `planner_kb/planner.py` — YAML prompt path
- `run_cr.py` — `task_file` path
- `evaluate.py` — `validation_dir` / `evaluate_dir`
