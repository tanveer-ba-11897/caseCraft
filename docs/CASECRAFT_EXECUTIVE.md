# CaseCraft Executive Summary

This is the concise technical stakeholder view of the project.

- For the detailed full documentation, see [README.md](../README.md).
- For the simple layman-friendly version, see [CASECRAFT_ELIF.md](./CASECRAFT_ELIF.md).

## What CaseCraft Does

CaseCraft converts feature requirements into structured QA test suites.

It combines:

- LLM-based test generation
- retrieval from product knowledge sources
- schema validation and output cleanup
- export to operational QA formats

The goal is to reduce manual test design effort while improving consistency, traceability, and edge-case coverage.

## How It Works

At a system level, the workflow is:

1. Ingest a feature document.
2. Parse and chunk the document.
3. Retrieve relevant product knowledge from the knowledge base.
4. Build structured prompts with feature context plus retrieved knowledge.
5. Generate candidate test cases using an LLM.
6. Sanitize, validate, normalize, deduplicate, and organize the output.
7. Export the final suite to Excel or JSON.

This keeps generation grounded in actual project knowledge rather than relying only on generic model behavior.

## Architecture Overview

CaseCraft is organized into three practical layers:

### 1. Interface Layer

This is the CLI entry point used to run generation and ingest knowledge.

Primary components:

- `cli/main.py` for test generation
- `cli/ingest.py` for knowledge ingestion

### 2. Orchestration Layer

This is the application core that coordinates parsing, retrieval, prompting, generation, and post-processing.

Primary components:

- `core/generator.py`
- `core/parser.py`
- `core/prompts.py`
- `core/config.py`
- `core/exporter.py`
- `core/llm_client.py`

### 3. Knowledge Layer

This is the retrieval system that improves context quality during generation.

Primary components:

- ChromaDB vector storage
- BM25 sparse retrieval
- optional knowledge graph traversal
- embedding and reranking models
- local and web document loaders

## Key Technical Capabilities

### Retrieval-Augmented Generation

CaseCraft uses a hybrid retrieval pipeline rather than pure prompting. It combines:

- dense vector search
- sparse keyword search
- score fusion
- optional reranking
- optional parent-child expansion
- optional knowledge graph expansion

This improves relevance and reduces generic output.

### Structured Output Generation

The system does not treat model output as trustworthy by default. It validates and repairs output before delivery through:

- JSON sanitation and repair
- schema enforcement with Pydantic
- field normalization
- duplicate removal
- reviewer-style refinement

### Knowledge Ingestion

CaseCraft can ingest:

- PDFs
- markdown files
- text files
- sitemap-based documentation
- individual URLs
- URL batches from files

This lets teams build a project-specific QA knowledge base over time.

## Operational Outputs

CaseCraft exports test suites in:

- Excel for direct QA team use
- JSON for automation, integration, or downstream processing

Each generated test case can include title, steps, preconditions, test data, expected results, tags, dependencies, and priority.

## Current Performance Shape

Runtime is dominated by model generation time.

In typical runs:

- parsing and retrieval are relatively small portions of total runtime
- LLM generation is the main bottleneck
- faster models and better prompt fit have the largest effect on turnaround time

This makes model selection and context sizing important operational decisions.

## Security and Reliability Considerations

CaseCraft includes practical safeguards relevant to enterprise use:

- path validation for file handling
- file type restrictions
- SSRF protections in web ingestion
- prompt-injection fencing heuristics
- integrity validation around persisted BM25 data
- structured error handling and output cleanup

These controls reduce the chance of unsafe or low-quality input destabilizing the workflow.

## Who This Is For

This project is relevant to:

- QA leads
- engineering managers
- product engineering teams
- technical program managers
- platform teams evaluating AI-assisted SDLC workflows

It is especially useful where documentation is growing faster than QA authoring capacity.

## Recommended Positioning

For internal adoption, CaseCraft should be positioned as:

- a QA productivity tool
- a consistency layer for test authoring
- a way to operationalize product documentation
- a foundation for defect-informed regression generation

It should not be positioned as a fully autonomous QA replacement.

## Next Strategic Phase

The next phase is to move from document-aware generation to feedback-aware generation.

Two high-value directions are:

1. Fine-tuning or instruction-tuning on accepted high-quality test suites to improve consistency and reduce repair overhead.
2. Ingesting historical defect data so the system can generate test coverage for known failure patterns and human-discovered edge cases.

This would strengthen regression quality and increase value as project history accumulates.

## Bottom Line

CaseCraft is an AI-assisted QA generation platform that turns product requirements and project knowledge into structured, reviewable test suites.

Its value is the combination of:

- project-aware retrieval
- structured QA output
- repeatable processing
- practical integration into testing workflows

That makes it a credible foundation for scaled, knowledge-driven test design.