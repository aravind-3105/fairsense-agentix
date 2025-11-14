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

from pathlib import Path
from typing import TypedDict

import pandas as pd
from langchain_core.documents import Document

from fairsense_agentix.configs import settings
from fairsense_agentix.tools.embeddings import LangChainEmbedder
from fairsense_agentix.tools.faiss_index import LangChainFAISSTool


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
    print(f"\n{'=' * 60}")
    print(f"Building index: {output_name}")
    print(f"{'=' * 60}")

    # Load CSV
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} documents")

    # Validate required column exists
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in {csv_path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert DataFrame to LangChain Documents
    print(f"Creating LangChain documents from column '{text_column}'...")
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

    print(f"  Created {len(documents)} documents")

    # Build FAISS index using LangChain (much simpler!)
    print("Building LangChain FAISS index...")
    print(f"  (Embedding {len(documents)} documents...)")

    faiss_tool = LangChainFAISSTool.from_documents(
        documents=documents,
        embeddings=embedder.embeddings,  # Access underlying LangChain embeddings
        metadata=metadata_list,
    )

    print(f"  ✓ Index built with {len(documents)} vectors")

    # Save index (LangChain format: folder with index.faiss + index.pkl)
    output_dir = Path("data/indexes")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving index to {output_dir}/{output_name}/...")
    faiss_tool.save_local(
        folder_path=output_dir,
        index_name=output_name,
    )

    print(f"  ✓ Saved index to {output_dir}/{output_name}/index.faiss")
    print(f"  ✓ Saved docstore to {output_dir}/{output_name}/index.pkl")
    print(f"  ✓ Saved metadata to {output_dir}/{output_name}_meta.json")

    print(f"✓ Successfully built {output_name} index!")


def main() -> None:
    """Build FAISS indexes for all configured datasets using LangChain.

    Phase 6.0: Updated to use LangChainEmbedder and LangChainFAISSTool for
    simpler index building with .from_documents() API.
    """
    print("\n" + "=" * 60)
    print("FAISS Index Builder (LangChain)")
    print("=" * 60)

    # Initialize LangChain embedder
    print(f"\nInitializing LangChain embedder: {settings.embedding_model}")
    embedder = LangChainEmbedder(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
        normalize=True,  # For cosine similarity
    )
    print(f"  ✓ Model loaded: {embedder.model_name}")
    print(f"  ✓ Dimension: {embedder.dimension}")

    # Define datasets to process
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

    # Build each index
    for dataset in datasets:
        try:
            build_index(
                csv_path=dataset["csv_path"],
                text_column=dataset["text_column"],
                output_name=dataset["output_name"],
                embedder=embedder,
            )
        except FileNotFoundError:
            print(f"\n⚠️  WARNING: {dataset['csv_path']} not found. Skipping...")
        except Exception as e:
            print(f"\n❌ ERROR building {dataset['output_name']}: {e}")
            raise

    # Summary
    print("\n" + "=" * 60)
    print("✓ All indexes built successfully!")
    print("=" * 60)
    print("\nBuilt indexes (LangChain format):")
    print("  - data/indexes/risks/index.faiss")
    print("  - data/indexes/risks/index.pkl")
    print("  - data/indexes/risks_meta.json")
    print("  - data/indexes/rmf/index.faiss")
    print("  - data/indexes/rmf/index.pkl")
    print("  - data/indexes/rmf_meta.json")
    print("\nYou can now use these indexes with LangChainFAISSTool!")
    print("Retriever pattern available: faiss_tool.as_retriever()")


if __name__ == "__main__":
    main()
