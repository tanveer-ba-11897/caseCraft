
import argparse
import logging
import re
import sys
import os
from pathlib import Path
from typing import Dict

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.generator import generate_test_suite
from core.exporter import export
from core.config import load_config


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI. DEBUG level when --verbose, INFO otherwise."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(
        description="CaseCraft: AI-powered test case generator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Load config to get defaults
    config = load_config()

    # Generate Command
    generate_parser = subparsers.add_parser("generate", help="Generate test cases from a document")
    generate_parser.add_argument("file", help="Path to the feature document (PDF, TXT, MD)")
    
    # We use 'None' as default so we know if user explicitly passed a flag, 
    # otherwise we fall back to config object.
    generate_parser.add_argument("--model", "-m", default=None, help=f"Ollama model to use (default: {config.general.model})")
    generate_parser.add_argument("--output", "-o", help="Output file path (default: [filename]_test_cases.xlsx)")
    generate_parser.add_argument("--format", "-f", choices=["excel", "json"], default=None, help=f"Output format (default: {config.output.default_format})")
    
    # Quality Flags - Tri-state logic (True/False/None)
    # Store true if --no-dedup passed, else None
    generate_parser.add_argument("--no-dedup-semantic", action="store_true", default=None, help=f"Disable semantic deduplication")
    # Store true if --reviewer passed, else None
    generate_parser.add_argument("--reviewer", "--quality", action="store_true", default=None, help=f"Enable AI reviewer pass (default: {config.quality.reviewer_pass})")
    
    # Feature configuration
    
    generate_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    # Add-Model Command
    add_model_parser = subparsers.add_parser(
        "add-model",
        help="Register an Ollama model in MODEL_SPECS (auto-detects context window and sets output cap)",
    )
    add_model_parser.add_argument("model", help="Ollama model name (e.g. llama3.2:1b, qwen2.5:7b)")
    add_model_parser.add_argument(
        "--max-output", type=int, default=None,
        help="Override the auto-detected max_output tokens (default: heuristic based on param count)",
    )
    add_model_parser.add_argument(
        "--context-window", type=int, default=None,
        help="Override the auto-detected context window (default: from Ollama /api/show)",
    )

    args = parser.parse_args()

    if args.command == "generate":
        _setup_logging(verbose=getattr(args, 'verbose', False))
        run_generate(args)
    elif args.command == "add-model":
        _setup_logging(verbose=False)
        run_add_model(args)
    else:
        parser.print_help()

def run_generate(args):
    logger = logging.getLogger("casecraft.cli")
    input_path = Path(args.file)
    if not input_path.exists():
        logger.error("File not found: %s", input_path)
        sys.exit(1)

    logger.info("Starting CaseCraft for %s", input_path.name)
    # Resolve config priorities: CLI Args > Config File/Env
    config = load_config()
    
    model = args.model if args.model else config.general.model
    fmt_str = args.format if args.format else config.output.default_format
    
    # Boolean logic is tricky with argparse 'store_true' vs config
    # If args.no_dedup_semantic is True (flag passed), we want dedup=False
    # If args.no_dedup_semantic is False/None (flag not passed), we check config
    if args.no_dedup_semantic: 
        dedup_semantic = False
    else:
        # If config says True, we want True.
        dedup_semantic = config.quality.semantic_deduplication
        
    reviewer_pass = args.reviewer if args.reviewer else config.quality.reviewer_pass

    # Determine output path if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        ext = "xlsx" if fmt_str == "excel" else "json"
        # Default to outputs/ folder in project root
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        output_path = outputs_dir / f"{input_path.stem}_test_cases.{ext}"

    try:
        logger.info("Parsing and generating test cases... (this may take a minute)")
        suite = generate_test_suite(
            str(input_path),
            model=model,
            dedup_semantic=dedup_semantic,
            reviewer_pass=reviewer_pass,
            app_type=None  # Falls back to config
        )
        
        logger.info("Generation complete. Found %d test cases.", len(suite.test_cases))
        
        logger.info("Exporting to %s...", output_path)
        export(suite, fmt_str, str(output_path))
        
        logger.info("Output saved to %s", output_path)
        
        # Unload the model from Ollama memory to free resources
        from core.llm_client import llm_client
        llm_client.unload_model(model)
        
    except Exception as e:
        logger.error("%s", e)
        if args.verbose:
            import traceback
            tb = traceback.format_exc()
            # Redact potential API keys / tokens from stack traces
            tb = re.sub(r'(?i)(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*\S+',
                        r'\1=<REDACTED>', tb)
            print(tb, file=sys.stderr)
        sys.exit(1)


def run_add_model(args):
    """Register a new Ollama model in MODEL_SPECS with auto-detected or user-supplied specs."""
    from core.llm_client import llm_client, MODEL_SPECS

    model = args.model
    logger = logging.getLogger("casecraft.cli")

    # Check if already registered
    if model in MODEL_SPECS and args.max_output is None and args.context_window is None:
        spec = MODEL_SPECS[model]
        print(f"Model '{model}' is already registered: context_window={spec['context_window']:,}, "
              f"max_output={spec['max_output']:,}")
        return

    # Auto-detect from Ollama
    try:
        spec = llm_client.auto_register_ollama_model(model)
    except RuntimeError as e:
        logger.error("%s", e)
        print(f"[ERROR] {e}")
        print("Make sure Ollama is running and the model is pulled.")
        sys.exit(1)

    # Apply user overrides
    if args.context_window is not None:
        spec["context_window"] = args.context_window
    if args.max_output is not None:
        spec["max_output"] = args.max_output
    MODEL_SPECS[model] = spec

    print(f"Registered '{model}' in MODEL_SPECS:")
    print(f"  context_window : {spec['context_window']:,} tokens")
    print(f"  max_output     : {spec['max_output']:,} tokens")

    # Persist to llm_client.py so the spec survives restarts
    _persist_model_spec(model, spec)


def _persist_model_spec(model: str, spec: Dict[str, int]) -> None:
    """Append or update a model entry in the MODEL_SPECS dict in llm_client.py."""
    llm_client_path = Path(__file__).parent.parent / "core" / "llm_client.py"
    source = llm_client_path.read_text(encoding="utf-8")

    # Check if model already has a line in source
    import re
    escaped_model = re.escape(f'"{model}"')
    pattern = re.compile(rf'^(\s*){escaped_model}\s*:\s*\{{.*\}},?\s*$', re.MULTILINE)
    new_line_content = (
        f'    "{model}": '
        f'{{"context_window": {spec["context_window"]:_}, "max_output": {spec["max_output"]:_}}},'
    )

    match = pattern.search(source)
    if match:
        # Update existing entry
        updated = source[:match.start()] + new_line_content + source[match.end():]
    else:
        # Insert before the closing brace of MODEL_SPECS
        # Find the "# Ollama local models" comment or the closing "}" of MODEL_SPECS
        ollama_marker = "    # Ollama local models\n"
        closing_brace_idx = source.find("\n}\n", source.find("MODEL_SPECS"))
        if closing_brace_idx == -1:
            print("[WARN] Could not locate MODEL_SPECS closing brace — runtime-only registration.")
            return

        insert_at = closing_brace_idx
        updated = source[:insert_at] + "\n" + new_line_content + source[insert_at:]

    llm_client_path.write_text(updated, encoding="utf-8")
    print(f"Persisted to {llm_client_path.relative_to(Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
