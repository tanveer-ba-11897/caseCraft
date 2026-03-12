
import logging
import os
import yaml
from typing import Annotated, Optional
from pathlib import Path
from pydantic import BaseModel, Field, SecretStr, ConfigDict

logger = logging.getLogger("casecraft.config")

# Constants
CONFIG_FILE_NAME = "casecraft.yaml"

class GeneralSettings(BaseModel):
    model_config = ConfigDict(validate_default=True)

    model: Annotated[str, Field(description="Ollama model to use")] = "llama3.1:8b"
    llm_provider: Annotated[str, Field(description="Backend provider: 'ollama', 'openai', 'google', or 'copilot'")] = "ollama"
    base_url: Annotated[str, Field(description="Base URL for LLM API")] = "http://localhost:11434"
    api_key: Annotated[SecretStr, Field(description="API Key (if required)")] = SecretStr("ollama")
    timeout: Annotated[int, Field(description="Request timeout in seconds")] = 600
    max_retries: Annotated[int, Field(description="Max retries for JSON generation")] = 2
    context_window_ratio: Annotated[float, Field(ge=0.0, le=1.0, description="Fraction of the model's native context window to use (0.0-1.0). 0.75 = use 75% of the detected window. The model's native window is auto-detected for Ollama; other providers default to 128K.")] = 0.75
    output_token_ratio: Annotated[float, Field(ge=0.0, le=1.0, description="(Deprecated — kept for backward compat) Static fallback ratio when prompt size is unknown. Dynamic allocation is preferred.")] = 0.25
    min_output_tokens: Annotated[int, Field(ge=256, description="Minimum guaranteed output tokens regardless of prompt size. Prevents the dynamic allocator from starving output when the prompt is very large.")] = 1024
    llm_call_delay: Annotated[float, Field(description="Minimum seconds between consecutive LLM API calls. Prevents rate-limit errors on providers like GitHub Models (copilot). 0 = no delay.")] = 0.0
    num_ctx: Annotated[int, Field(ge=0, description="Ollama num_ctx override. 0 = use full effective context window (native × context_window_ratio). Positive value = pass this exact num_ctx to Ollama. Use when auto-sizing causes issues.")] = 0

class GenerationSettings(BaseModel):
    chunk_size: Annotated[int, Field(description="Document chunk size")] = 1000
    chunk_overlap: Annotated[int, Field(description="Chunk overlap")] = 100
    temperature: Annotated[float, Field(description="Model temperature")] = 0.2
    top_p: Annotated[float, Field(description="Nucleus sampling (creativity range)")] = 0.9
    app_type: Annotated[str, Field(description="Application type: web, mobile, desktop, api")] = "web"
    min_cases_per_chunk: Annotated[int, Field(description="Minimum test cases expected per chunk. Triggers re-generation if under this count.")] = 5
    max_workers: Annotated[int, Field(description="Maximum parallel threads for chunk generation. Set to 1 for rate-limited providers to avoid concurrent API calls.")] = 4
    skip_condensation: Annotated[bool, Field(description="Skip the condensation LLM call and send raw chunks directly. Halves the LLM calls, best for small/local models where condensation is slower than just sending the full chunk.")] = False
    max_cases_per_chunk: Annotated[int, Field(ge=0, description="Maximum test cases to generate per chunk. The LLM is instructed to stop after this many. 0 = unlimited (model decides). Lower values = faster generation on CPU (fewer output tokens). 3-5 is recommended for local models.")] = 0
    max_kb_ratio: Annotated[int, Field(ge=1, description="Maximum KB context size as a multiple of the feature chunk size. Caps KB context to N× the (condensed) chunk length to avoid prompt bloat. Higher = more context but slower; lower = faster but less KB grounding.")] = 3

class OutputSettings(BaseModel):
    default_format: Annotated[str, Field(description="Default output format (excel/json)")] = "excel"
    output_dir: Annotated[str, Field(description="Default output directory")] = "outputs"

class KnowledgeSettings(BaseModel):
    vector_db_path: Annotated[str, Field(description="ChromaDB persistence directory for the vector store")] = "knowledge_base/chroma_db"
    kb_chunk_size: Annotated[int, Field(description="Max characters per knowledge base chunk during ingestion")] = 1500
    min_score_threshold: Annotated[float, Field(description="Minimum hybrid score to include a retrieval result (0.0 - 1.0). Lower = more results.")] = 0.1
    query_decomposition: Annotated[bool, Field(description="Decompose long queries into focused sub-queries for better retrieval")] = True
    query_expansion: Annotated[bool, Field(description="Expand queries with domain synonyms for improved recall")] = True
    max_sub_queries: Annotated[int, Field(description="Maximum sub-queries when decomposition is enabled")] = 4
    parent_child_chunking: Annotated[bool, Field(description="Enable two-tier parent-child chunking. Small child chunks are used for retrieval precision; parent chunks are returned for richer context.")] = False
    child_chunk_size: Annotated[int, Field(description="Max characters per child chunk when parent-child chunking is enabled.")] = 400
    child_overlap: Annotated[int, Field(description="Character overlap between adjacent child chunks of the same parent.")] = 50
    knowledge_graph: Annotated[bool, Field(description="Build a lightweight relation graph over chunks at ingest time. Enables graph-expanded retrieval for cross-document tracing.")] = False
    graph_path: Annotated[str, Field(description="Path for the persisted knowledge graph JSON file.")] = "knowledge_base/knowledge_graph.json"
    graph_max_hops: Annotated[int, Field(description="Maximum BFS hops when expanding retrieval results via the knowledge graph.")] = 2
    graph_max_expansion: Annotated[int, Field(description="Maximum number of graph-expanded chunks to append to retrieval results.")] = 3

class QualitySettings(BaseModel):
    semantic_deduplication: Annotated[bool, Field(description="Enable semantic deduplication")] = True
    similarity_threshold: Annotated[float, Field(description="Similarity threshold for deduplication")] = 0.85
    reviewer_pass: Annotated[bool, Field(description="Enable AI reviewer pass")] = True
    top_k: Annotated[int, Field(description="Number of context chunks to retrieve. Set to -1 to retrieve ALL chunks.")] = 5
    max_kb_batches: Annotated[int, Field(description="Max KB context batches to process per feature chunk. Batches are ordered by relevance, so this keeps only the most useful context. Set to -1 for unlimited.")] = 10
    shared_kb_retrieval: Annotated[bool, Field(description="Use a single KB retrieval for the whole document instead of per-chunk. Faster but less targeted context.")] = False

class CacheSettings(BaseModel):
    persist_bm25_index: Annotated[bool, Field(description="Persist BM25 sparse index to disk. Skips rebuild on startup when KB is unchanged.")] = True

class CaseCraftConfig(BaseModel):
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)

def load_config(config_path: Optional[str] = None) -> CaseCraftConfig:
    """
    Load configuration from file and environment variables.
    Priority: CLI (passed args) > Env Vars > Config File > Defaults
    """
    # 1. Determine config file path
    if config_path:
        path = Path(config_path)
    else:
        # Use project root (parent of core/) to avoid loading untrusted configs from CWD
        project_root = Path(__file__).parent.parent
        path = project_root / CONFIG_FILE_NAME

    config_data = {}

    # 2. Load from YAML if exists
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Failed to load config file: %s", e)

    # Warn if API key is stored in config file instead of env var
    yaml_api_key = config_data.get("general", {}).get("api_key")
    if yaml_api_key and yaml_api_key != "ollama":
        logger.warning("API key detected in config file. Use env var CASECRAFT_GENERAL_API_KEY instead for security.")

    # 3. Load from Environment Variables (Override YAML)
    # Convention: CASECRAFT_SECTION_KEY (e.g., CASECRAFT_GENERAL_MODEL)
    
    sections = {
        "general": GeneralSettings,
        "generation": GenerationSettings,
        "output": OutputSettings,
        "quality": QualitySettings,
        "knowledge": KnowledgeSettings,
        "cache": CacheSettings,
    }

    for section_name, section_model in sections.items():
        if section_name not in config_data:
            config_data[section_name] = {}
            
        for field_name in section_model.model_fields.keys():
            env_var = f"CASECRAFT_{section_name.upper()}_{field_name.upper()}"
            env_val = os.getenv(env_var)
            if env_val is not None:
                # Simple type conversion
                if env_val.lower() == "true":
                    val = True
                elif env_val.lower() == "false":
                    val = False
                else:
                    try:
                        # Try int/float
                        if "." in env_val:
                            val = float(env_val)
                        else:
                            val = int(env_val)
                    except ValueError:
                        val = env_val
                
                config_data[section_name][field_name] = val

    # 4. Create and return Pydantic model
    return CaseCraftConfig(**config_data)

# Global config instance (loaded lazily or explicitly)
config = load_config()
