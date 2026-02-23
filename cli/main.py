
import argparse
import sys
import os
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.generator import generate_test_suite
from core.exporter import export
from core.config import load_config

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

    args = parser.parse_args()

    if args.command == "generate":
        run_generate(args)
    else:
        parser.print_help()

def run_generate(args):
    input_path = Path(args.file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    print(f"[CaseCraft] Starting for {input_path.name}")
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
        print("[INFO] Parsing and generating test cases... (this may take a minute)")
        suite = generate_test_suite(
            str(input_path),
            model=model,
            dedup_semantic=dedup_semantic,
            reviewer_pass=reviewer_pass,
            app_type=None  # Falls back to config
        )
        
        print(f"[SUCCESS] Generation complete. Found {len(suite.test_cases)} test cases.")
        
        print(f"[INFO] Exporting to {output_path}...")
        export(suite, fmt_str, str(output_path))
        
        print(f"[DONE] Output saved to {output_path}.")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
