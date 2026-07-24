"""FastAPI request models. Examples are deliberately generic (app-agnostic)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .config import APP_NAME


class ChatRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "project": "my-app",
        "prompt": "What validation rules apply to the primary required input field?",
        "top_k": 3,
        "max_new_tokens": 512,
        "enable_thinking": False,
    }})

    prompt: str = Field(..., min_length=1, description="Question or instruction for the LLM, automatically augmented with relevant SRS context retrieved from the knowledge graph.")
    project: str = Field(default="default", description="Project identifier used to scope knowledge-graph retrieval.")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of SRS chunks to retrieve as RAG context.")
    max_new_tokens: int = Field(default=2048, ge=64, le=8192, description="Maximum tokens in the LLM response.")
    enable_thinking: bool = Field(default=False, description="Enable extended chain-of-thought reasoning (supported models only).")


class NextTestCaseRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "project": "my-app",
        "app_name": "the app under test",
        "objective": "generate next high-value non-duplicate exploratory test case",
        "top_k": 8,
        "max_new_tokens": 4096,
        "enable_thinking": False,
        "debug_trace": False,
        "max_retrieval_rounds": 3,
    }})

    project: str = Field(..., min_length=1, description="Project identifier scoping the Neo4j knowledge graph.")
    app_name: str = Field(default=APP_NAME, description="Display name of the app under test, injected into every LLM prompt.")
    objective: str = Field(
        default="generate the next best high-value non-duplicate exploratory test case",
        description="High-level exploration goal for this round. The planner adapts automatically based on live coverage state — override only for targeted sessions.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Max SRS chunks retrieved per knowledge-graph query during the retrieval planning loop.")
    max_new_tokens: int = Field(default=2048, ge=64, le=8192, description="Token budget for the final test case generation call.")
    enable_thinking: bool = Field(default=False, description="Enable extended reasoning traces (supported models only). Increases latency.")
    debug_trace: bool = Field(default=False, description="Include full debug trace in the response: prompt texts, raw model output, and retrieved context blocks for every planning round.")
    max_retrieval_rounds: int = Field(default=3, ge=1, le=6, description="Maximum planning/retrieval rounds before the gateway forces test case generation. Higher values gather more context but increase latency.")


class LogVerdictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "project": "my-app",
        "app_name": "the app under test",
        "test_case_id": "TC-003",
        "title": "Verify the primary input field rejects malformed data on the entry screen",
        "verdict": "failed",
        "notes": "App accepted an invalid value without showing a validation error.",
        "area": "data_entry",
        "requirement_ids": ["FR-7"],
        "next_objective": "",
        "top_k": 8,
        "max_new_tokens": 4096,
        "enable_thinking": False,
        "debug_trace": False,
    }})

    project: str = Field(..., min_length=1, description="Project identifier.")
    app_name: str = Field(default=APP_NAME, description="Display name of the app under test.")
    test_case_id: str = Field(..., min_length=1, description="ID of the executed test case (e.g. 'TC-003'). Must match the value returned by the planner.")
    title: str = Field(..., min_length=1, description="Title of the executed test case.")
    verdict: str = Field(
        ...,
        pattern="^(pass|failed|blocked|skipped)$",
        description=(
            "Execution outcome. Accepted values:\n"
            "- `pass` — test ran successfully, expected result met.\n"
            "- `failed` — test ran but expected result was NOT met (bug found).\n"
            "- `blocked` — test could not run due to a prerequisite or environment issue.\n"
            "- `skipped` — test was intentionally skipped.\n\n"
            "`blocked` and `skipped` are stored as `failed` in the knowledge graph with the original verdict prepended to notes."
        ),
    )
    notes: str = Field(default="", description="Execution notes, error messages, or observations from the executor. Stored in the knowledge graph and used to inform subsequent test generation.")
    area: str = Field(default="general", description="Feature area of the test case (a runtime-derived slug, e.g. 'data_entry', 'search'). Used to compute the coverage map and drive the adaptive next-test objective.")
    requirement_ids: list[str] = Field(default_factory=list, description="Requirement IDs (e.g. ['FR-5','FR-7']) this test exercises. Creates COVERS edges in the knowledge graph for graph-native requirement coverage.")
    next_objective: str = Field(default="", description="Override the adaptive objective for the next test case. Leave blank to let the planner derive the objective from the verdict and coverage state (recommended).")
    top_k: int = Field(default=5, ge=1, le=20, description="Max SRS chunks retrieved for the next test case generation.")
    max_new_tokens: int = Field(default=2048, ge=64, le=8192, description="Token budget for the next test case generation call.")
    enable_thinking: bool = Field(default=False, description="Enable extended reasoning for the next test generation call.")
    debug_trace: bool = Field(default=False, description="Include full debug trace in the next test case response.")


class IngestSRSRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "project": "my-app",
        "source_path": "./data/inputs/Sample-Contacts-App-SRS.txt",
        "chunk_chars": 1200,
        "use_model_summary": True,
        "require_model_summary": True,
        "extract_entities": True,
    }})

    project: str = Field(..., min_length=1, description="Project identifier. Created automatically if it does not exist.")
    source_path: str = Field(..., min_length=1, description="Local path to the SRS document (.txt/.md/.pdf/.docx/.html/...). Any format is auto-converted to text. Path traversal ('..') is rejected. If `srs_text` is provided this field is used as a label only.")
    srs_text: str | None = Field(default=None, description="Inline SRS text. If provided, the file at `source_path` is not read. Maximum 500,000 characters.")
    chunk_chars: int = Field(default=1200, ge=200, le=5000, description="Target character size for each SRS chunk stored in the knowledge graph. Smaller chunks improve retrieval precision; larger chunks improve coherence.")
    use_model_summary: bool = Field(default=True, description="Generate a planner-friendly SRS summary via the LLM before ingesting chunks. The summary is used in every test generation call for global context.")
    require_model_summary: bool = Field(default=True, description="If true, the request returns 503 when model summarization fails. If false, falls back to a rule-based keyword summary.")
    extract_entities: bool = Field(default=True, description="Run LLM structured extraction to build the requirement entity graph (Requirement/Entity/ValidationRule nodes). Falls back to rule-based extraction if the model is unavailable.")


class IngestFigmaRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "project": "my-app",
        "source_path": "./data/inputs/GENERATED_JSON.json",
        "use_model_classification": True,
    }})

    project: str = Field(..., min_length=1, description="Project identifier. Created automatically if it does not exist.")
    source_path: str = Field(..., min_length=1, description="Local path to the exported Figma JSON file. Path traversal ('..') is rejected. If `figma_json` is provided this field is used as a label only.")
    figma_json: str | None = Field(default=None, description="Inline Figma export JSON string. Accepts raw JSON or a markdown code-fenced JSON block.")
    use_model_classification: bool = Field(default=True, description="Use the LLM to classify each screen into a feature-area purpose (dynamic, app-agnostic) instead of relying on the screen name slug. Falls back to the name-derived slug if the model is unavailable.")


class ResetProjectRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "project": "my-app",
        "delete_tests": True,
        "delete_srs": False,
        "delete_figma": False,
    }})

    project: str = Field(..., min_length=1, description="Project to reset.")
    delete_tests: bool = Field(default=True, description="Delete all logged test cases and test runs. The next test generation starts from scratch with no history.")
    delete_srs: bool = Field(default=False, description="Delete ingested SRS chunks, summary, and requirement entity graph. Re-ingest via `/srs/ingest` before the next test generation.")
    delete_figma: bool = Field(default=False, description="Delete ingested Figma screens and UI elements. Re-ingest via `/figma/ingest` before the next test generation.")
