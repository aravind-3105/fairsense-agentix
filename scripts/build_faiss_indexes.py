#!/usr/bin/env python
"""Build FAISS indexes from CSV datasets.

This script reads AI risks and RMF datasets from CSV files, generates embeddings,
and builds FAISS indexes for semantic search.

Usage
-----
    python scripts/build_faiss_indexes.py

Requirements
------------
- data/raw/ai_risks.csv
- data/raw/ai_rmf.csv

Outputs
-------
- data/indexes/risks.faiss
- data/indexes/risks_meta.json
- data/indexes/rmf.faiss
- data/indexes/rmf_meta.json
"""

import json
from pathlib import Path
from typing import TypedDict

import faiss
import numpy as np
import pandas as pd

from fairsense_agentix.configs import settings
from fairsense_agentix.tools.embeddings import SentenceTransformerEmbedder


class DatasetConfig(TypedDict):
    """Configuration for a dataset to index."""

    csv_path: Path
    text_column: str
    output_name: str


def build_index(
    csv_path: Path,
    text_column: str,
    output_name: str,
    embedder: SentenceTransformerEmbedder,
) -> None:
    """Build FAISS index from CSV dataset.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file
    text_column : str
        Name of column containing text to embed
    output_name : str
        Base name for output files (e.g., "risks" → "risks.faiss")
    embedder : SentenceTransformerEmbedder
        Embedder for generating vectors
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

    # Generate embeddings
    print(f"Generating embeddings for column '{text_column}'...")
    texts = df[text_column].tolist()
    vectors = []

    for i, text in enumerate(texts):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(texts)}")

        vector = embedder.encode(str(text))
        vectors.append(vector)

    vectors_array = np.array(vectors, dtype="float32")
    print(f"  Generated {len(vectors)} vectors of dimension {vectors_array.shape[1]}")

    # Build FAISS index (IndexFlatIP for inner product / cosine similarity)
    print("Building FAISS index...")
    dimension = embedder.dimension
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors_array)
    print(f"  Index built with {index.ntotal} vectors")

    # Save index
    index_path = Path("data/indexes") / f"{output_name}.faiss"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"  Saved index to {index_path}")

    # Prepare metadata (all columns as JSON-serializable dicts)
    print("Preparing metadata...")
    metadata = df.to_dict(orient="records")

    # Save metadata
    metadata_path = Path("data/indexes") / f"{output_name}_meta.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")

    print(f"✓ Successfully built {output_name} index!")


def main() -> None:
    """Build FAISS indexes for all configured datasets."""
    print("\n" + "=" * 60)
    print("FAISS Index Builder")
    print("=" * 60)

    # Initialize embedder
    print(f"\nInitializing embedder: {settings.embedding_model}")
    embedder = SentenceTransformerEmbedder(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )
    print(f"  Model loaded: {embedder.model_name}")
    print(f"  Dimension: {embedder.dimension}")

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
    print("\nBuilt indexes:")
    print("  - data/indexes/risks.faiss")
    print("  - data/indexes/risks_meta.json")
    print("  - data/indexes/rmf.faiss")
    print("  - data/indexes/rmf_meta.json")
    print("\nYou can now use these indexes with FAISSIndexTool!")


if __name__ == "__main__":
    main()
