#!/usr/bin/env python
"""Build FAISS indexes from CSV datasets using LangChain.

Phase 6.0: Updated to use LangChain's FAISS vector store for simpler index
building and better integration with retriever patterns.

This script reads AI risks and RMF datasets from CSV files, generates embeddings,
and builds FAISS indexes for semantic search using LangChain's .from_documents() API.

Usage
-----
    python scripts/build_faiss_indexes.py

Requirements
------------
- data/raw/ai_risks.csv
- data/raw/ai_rmf.csv

Outputs (LangChain format)
-------
- data/indexes/risks/index.faiss
- data/indexes/risks/index.pkl
- data/indexes/risks_meta.json
- data/indexes/rmf/index.faiss
- data/indexes/rmf/index.pkl
- data/indexes/rmf_meta.json
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import pandas as pd
from langchain_core.documents import Document


if TYPE_CHECKING:
    from fairsense_agentix.tools.embeddings import LangChainEmbedder

logger = logging.getLogger(__name__)


def _load_runtime_settings() -> Any:
    """Load the Settings singleton without importing package ``__init__``.

    Importing ``fairsense_agentix.configs.settings`` normally runs
    ``fairsense_agentix/__init__.py``, which pulls heavy deps (e.g. transformers)
    even when this script exits early.
    """
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "fairsense_agentix" / "configs" / "settings.py"
    spec = importlib.util.spec_from_file_location(
        "fairsense_agentix_configs_settings_cli",
        path,
    )
    if spec is None or spec.loader is None:
        msg = f"Cannot load settings module from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.settings


class DatasetConfig(TypedDict):
    """Configuration for a dataset to index."""

    csv_path: Path
    text_column: str
    output_name: str


def build_index(
    csv_path: Path,
    text_column: str,
    output_name: str,
    embedder: LangChainEmbedder,
) -> None:
    """Build FAISS index from CSV dataset using LangChain.

    Phase 6.0: Simplified using LangChain's .from_documents() API instead of
    manual numpy/faiss operations.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file
    text_column : str
        Name of column containing text to embed
    output_name : str
        Base name for output files (e.g., "risks" → "risks/")
    embedder : LangChainEmbedder
        LangChain embedder for generating vectors
    """
    logger.info("")
    logger.info("%s", "=" * 60)
    logger.info("Building index: %s", output_name)
    logger.info("%s", "=" * 60)

    # Load CSV
    logger.info("Loading %s...", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("  Loaded %s documents", len(df))

    # Validate required column exists
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in {csv_path}. "
            f"Available columns: {df.columns.tolist()}",
        )

    # Convert DataFrame to LangChain Documents
    logger.info("Creating LangChain documents from column '%s'...", text_column)
    documents = []
    metadata_list = []

    for _, row in df.iterrows():
        # Extract text content
        text_content = str(row[text_column])

        # Prepare metadata (all columns except text column)
        metadata = row.to_dict()
        metadata_list.append(metadata)

        # Create LangChain Document
        doc = Document(
            page_content=text_content,
            metadata=metadata,
        )
        documents.append(doc)

    logger.info("  Created %s documents", len(documents))

    # Build FAISS index using LangChain (much simpler!)
    logger.info("Building LangChain FAISS index...")
    logger.info("  (Embedding %s documents...)", len(documents))

    from fairsense_agentix.tools.faiss_index import LangChainFAISSTool  # noqa: PLC0415

    faiss_tool = LangChainFAISSTool.from_documents(
        documents=documents,
        embeddings=embedder.embeddings,  # Access underlying LangChain embeddings
        metadata=metadata_list,
    )

    logger.info("  Index built with %s vectors", len(documents))

    # Save index (LangChain format: folder with index.faiss + index.pkl)
    output_dir = Path("data/indexes")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving index to %s/%s/...", output_dir, output_name)
    faiss_tool.save_local(
        folder_path=output_dir,
        index_name=output_name,
    )

    logger.info("  Saved index to %s/%s/index.faiss", output_dir, output_name)
    logger.info("  Saved docstore to %s/%s/index.pkl", output_dir, output_name)
    logger.info("  Saved metadata to %s/%s_meta.json", output_dir, output_name)

    logger.info("Successfully built %s index.", output_name)


def _artifact_lines_for_index(output_name: str) -> list[str]:
    """Return expected output paths for a built index (LangChain layout)."""
    base = Path("data/indexes")
    return [
        str(base / output_name / "index.faiss"),
        str(base / output_name / "index.pkl"),
        str(base / f"{output_name}_meta.json"),
    ]


def main() -> int:
    """Build FAISS indexes for all configured datasets using LangChain.

    Phase 6.0: Updated to use LangChainEmbedder and LangChainFAISSTool for
    simpler index building with .from_documents() API.

    Returns
    -------
    int
        0 if at least one index was built, 1 if nothing was built or inputs missing.
    """
    level_name = os.environ.get("LOGLEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    print("\n" + "=" * 60)
    print("FAISS Index Builder (LangChain)")
    print("=" * 60)

    datasets: list[DatasetConfig] = [
        {
            "csv_path": Path("data/raw/ai_risks.csv"),
            "text_column": "description",
            "output_name": "risks",
        },
        {
            "csv_path": Path("data/raw/ai_rmf.csv"),
            "text_column": "action",
            "output_name": "rmf",
        },
    ]

    available = [d for d in datasets if d["csv_path"].is_file()]
    if not available:
        logger.error("No input CSVs found. Expected at least one of:")
        for d in datasets:
            logger.error("  - %s", d["csv_path"])
        logger.error("Generate them (e.g. scripts/transform_mit_data.py) and retry.")
        return 1

    # Defer settings + embedder until CSVs exist (avoids package __init__ / ST).
    settings = _load_runtime_settings()
    from fairsense_agentix.tools.embeddings import LangChainEmbedder  # noqa: PLC0415

    logger.info("")
    logger.info("Initializing LangChain embedder: %s", settings.embedding_model)
    embedder = LangChainEmbedder(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
        normalize=True,  # For cosine similarity
    )
    logger.info("  Model loaded: %s", embedder.model_name)
    logger.info("  Dimension: %s", embedder.dimension)

    indexes_built: list[str] = []

    for dataset in datasets:
        if not dataset["csv_path"].is_file():
            logger.warning("%s not found. Skipping.", dataset["csv_path"])
            continue
        try:
            build_index(
                csv_path=dataset["csv_path"],
                text_column=dataset["text_column"],
                output_name=dataset["output_name"],
                embedder=embedder,
            )
        except Exception as e:
            logger.error("Error building %s: %s", dataset["output_name"], e)
            raise
        indexes_built.append(dataset["output_name"])

    if not indexes_built:
        logger.error("No indexes were built.")
        return 1

    logger.info("")
    logger.info("%s", "=" * 60)
    logger.info(
        "Finished building %s index(es): %s",
        len(indexes_built),
        ", ".join(indexes_built),
    )
    logger.info("%s", "=" * 60)
    logger.info("")
    logger.info("Output files (LangChain format):")
    for name in indexes_built:
        for line in _artifact_lines_for_index(name):
            logger.info("  - %s", line)
    logger.info("")
    logger.info("You can use these indexes with LangChainFAISSTool.")
    logger.info("Retriever pattern: faiss_tool.as_retriever()")

    return 0


if __name__ == "__main__":
    sys.exit(main())
