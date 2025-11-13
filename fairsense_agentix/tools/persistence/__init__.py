"""Persistence tools for FairSense-AgentiX.

This module provides file persistence tools for saving analysis results
to disk in various formats (CSV, JSON).

Examples
--------
    >>> from fairsense_agentix.tools.persistence import CSVWriter
    >>> from pathlib import Path
    >>>
    >>> writer = CSVWriter(output_dir=Path("outputs"))
    >>>
    >>> # Save CSV file
    >>> data = [
    ...     {"risk": "Bias", "severity": "High", "score": 0.89},
    ...     {"risk": "Privacy", "severity": "Medium", "score": 0.65},
    ... ]
    >>> csv_path = writer.save_csv(data, filename="risks_report.csv")
    >>> print(csv_path)
    /absolute/path/to/outputs/risks_report.csv
    >>>
    >>> # Save JSON file
    >>> json_path = writer.save_json(data, filename="risks_report.json")
"""

from fairsense_agentix.tools.persistence.file_writer import CSVWriter


__all__ = ["CSVWriter"]
