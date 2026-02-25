
import os
import yaml
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field, SecretStr, ConfigDict

# Constants
CONFIG_FILE_NAME = "casecraft.yaml"

class GeneralSettings(BaseModel):
    model_config = ConfigDict(validate_default=True)

    model: str = Field("llama3.1:8b", description="Ollama model to use")
    llm_provider: str = Field("ollama", description="Backend provider: 'ollama' or 'openai'")
    base_url: str = Field("http://localhost:11434", description="Base URL for LLM API")
    api_key: SecretStr = Field(default="ollama", description="API Key (if required)")
    timeout: int = Field(600, description="Request timeout in seconds")
    max_retries: int = Field(2, description="Max retries for JSON generation")
    context_window_size: int = Field(-1, description="Context window size (num_ctx). Set to -1 to use model default/max.")
    max_output_tokens: int = Field(4096, description="Max tokens to generate")

class GenerationSettings(BaseModel):
    chunk_size: int = Field(1000, description="Document chunk size")
    chunk_overlap: int = Field(100, description="Chunk overlap")
    temperature: float = Field(0.2, description="Model temperature")
    top_p: float = Field(0.5, description="Nucleus sampling (creativity range)")
    app_type: str = Field("web", description="Application type: web, mobile, desktop, api")

class OutputSettings(BaseModel):
    default_format: str = Field("excel", description="Default output format (excel/json)")
    output_dir: str = Field("outputs", description="Default output directory")

class KnowledgeSettings(BaseModel):
    kb_chunk_size: int = Field(1500, description="Max characters per knowledge base chunk during ingestion")
    min_score_threshold: float = Field(0.1, description="Minimum hybrid score to include a retrieval result (0.0 - 1.0). Lower = more results.")

class QualitySettings(BaseModel):
    semantic_deduplication: bool = Field(True, description="Enable semantic deduplication")
    similarity_threshold: float = Field(0.85, description="Similarity threshold for deduplication")
    reviewer_pass: bool = Field(False, description="Enable AI reviewer pass")
    top_k: int = Field(5, description="Number of context chunks to retrieve. Set to -1 to retrieve ALL chunks.")

class CaseCraftConfig(BaseModel):
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)

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
            print(f"Warning: Failed to load config file: {e}")

    # Warn if API key is stored in config file instead of env var
    yaml_api_key = config_data.get("general", {}).get("api_key")
    if yaml_api_key and yaml_api_key != "ollama":
        print("WARNING: API key detected in config file. Use env var CASECRAFT_GENERAL_API_KEY instead for security.")

    # 3. Load from Environment Variables (Override YAML)
    # Convention: CASECRAFT_SECTION_KEY (e.g., CASECRAFT_GENERAL_MODEL)
    
    sections = {
        "general": GeneralSettings,
        "generation": GenerationSettings,
        "output": OutputSettings,
        "quality": QualitySettings,
        "knowledge": KnowledgeSettings,
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
