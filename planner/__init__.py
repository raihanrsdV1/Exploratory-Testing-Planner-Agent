"""
Exploratory-testing planner package.

The monolithic gateway is decomposed into focused, app-agnostic modules:

    config            env configuration + gateway auth
    model_client      LLM backends (ngrok / gemini / openrouter)
    rag_client        knowledge-graph (RAG API) HTTP helpers
    textutil          pure JSON / similarity / query helpers
    coverage          live exploration coverage map + directives
    context_builders  format SRS/UI/history into compact prompt blocks
    prompts           all LLM prompt builders (exploratory-testing focused)
    schemas           FastAPI request models
    pipeline          orchestration (ingest, retrieve, generate, log, coverage)

`gateway/main.py` is now a thin FastAPI router that delegates to
`planner.pipeline`. Nothing here is specific to any single app — all domain
knowledge comes from the ingested SRS/UI graph at runtime.
"""
