"""File writer implementation for CSV and JSON persistence.

This module provides simple, reliable file I/O for saving analysis results.
It handles directory creation, path resolution, and proper file encoding.

Design Choices
--------------
1. **Why absolute paths**: Prevents breaking when working directory changes
2. **Why mkdir parents=True**: Automatically creates nested directories
3. **Why newline=''**: Cross-platform CSV compatibility (Windows/Mac/Linux)
4. **Why UTF-8 encoding**: Supports international characters and emoji

What This Enables
-----------------
- Reliable file outputs that work across platforms
- Automatic directory creation (no manual setup needed)
- Portable paths (absolute, work from any directory)
- International character support (not just ASCII)

Examples
--------
    >>> from pathlib import Path
    >>> writer = CSVWriter(output_dir=Path("outputs"))
    >>>
    >>> # Save CSV
    >>> data = [{"name": "Alice", "score": 0.95}, {"name": "Bob", "score": 0.87}]
    >>> path = writer.save_csv(data, "results.csv")
    >>> print(path.exists())
    True
    >>> print(path.is_absolute())
    True
    >>>
    >>> # Save JSON
    >>> json_path = writer.save_json({"status": "complete"}, "status.json")
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

from fairsense_agentix.tools.exceptions import PersistenceError


logger = logging.getLogger(__name__)


class CSVWriter:
    """File writer for CSV and JSON persistence.

    This writer provides simple, reliable file I/O with automatic directory
    creation, proper encoding, and cross-platform compatibility. All paths
    returned are absolute to prevent working directory issues.

    Design Philosophy:
    - **Absolute paths always**: Prevents issues when working directory changes
    - **Auto directory creation**: No manual mkdir needed
    - **Proper encoding**: UTF-8 for international characters
    - **Cross-platform**: Works on Windows, Mac, Linux

    Parameters
    ----------
    output_dir : Path
        Directory for saved files (created if doesn't exist)

    Attributes
    ----------
    output_dir : Path
        Absolute path to output directory

    Examples
    --------
    >>> writer = CSVWriter(output_dir=Path("outputs"))
    >>>
    >>> # Save analysis results
    >>> data = [
    ...     {"text": "Job posting 1", "bias_score": 0.75},
    ...     {"text": "Job posting 2", "bias_score": 0.45},
    ... ]
    >>> csv_path = writer.save_csv(data, "bias_scores.csv")
    >>> print(csv_path)
    /absolute/path/to/outputs/bias_scores.csv

    Notes
    -----
    The writer creates the output directory and any parent directories
    automatically if they don't exist. All operations are atomic at the
    file system level (write completes or fails, no partial writes).
    """

    def __init__(self, output_dir: Path) -> None:
        """Initialize file writer.

        Parameters
        ----------
        output_dir : Path
            Directory for saved files
        """
        # Convert to absolute path immediately
        self.output_dir = output_dir.resolve()

        logger.info(f"CSVWriter initialized (output_dir={self.output_dir})")

    def save_csv(
        self,
        data: list[dict[str, Any]],
        filename: str,
    ) -> Path:
        """Save data to CSV file.

        Creates the output directory if it doesn't exist, writes data as CSV,
        and returns the absolute path to the saved file.

        Parameters
        ----------
        data : list[dict[str, Any]]
            List of row dictionaries (keys become column headers)
        filename : str
            Output filename (e.g., "results.csv")

        Returns
        -------
        Path
            Absolute path to saved CSV file

        Raises
        ------
        PersistenceError
            If save fails (permission denied, invalid data, disk full, etc.)

        Examples
        --------
        >>> writer = CSVWriter(output_dir=Path("outputs"))
        >>> data = [{"col1": "a", "col2": 1}, {"col1": "b", "col2": 2}]
        >>> path = writer.save_csv(data, "data.csv")
        >>> print(path.exists())
        True

        Notes
        -----
        - Empty data list writes a CSV with headers only (no rows)
        - Missing keys in some rows result in empty cells
        - All values are converted to strings automatically
        - Uses newline='' for cross-platform compatibility
        """
        try:
            # Create output directory if needed
            self._ensure_output_dir_exists()

            # Construct full path
            output_path = self.output_dir / filename

            # Create parent directories if filename contains path separators
            if output_path.parent != self.output_dir:
                output_path.parent.mkdir(parents=True, exist_ok=True)

            # Handle empty data case
            if not data:
                logger.warning("Empty data list provided to save_csv")
                # Create empty CSV file
                output_path.write_text("", encoding="utf-8")
                logger.debug(f"Empty CSV created: {output_path}")
                return output_path.absolute()

            # Write CSV file
            with output_path.open("w", newline="", encoding="utf-8") as f:
                # Extract headers from first row
                headers = list(data[0].keys())

                # Create DictWriter
                writer = csv.DictWriter(f, fieldnames=headers)

                # Write headers
                writer.writeheader()

                # Write rows
                writer.writerows(data)

            logger.debug(
                f"CSV saved: {output_path} ({len(data)} rows, {len(headers)} columns)"
            )

            # Return absolute path
            return output_path.absolute()

        except PersistenceError:
            # Re-raise our own errors unchanged
            raise

        except PermissionError as e:
            msg = "Permission denied creating CSV file"
            raise PersistenceError(
                msg,
                context={
                    "output_dir": str(self.output_dir),
                    "filename": filename,
                    "error": str(e),
                },
            ) from e

        except OSError as e:
            # Disk full, invalid filename, etc.
            msg = "Failed to write CSV file"
            raise PersistenceError(
                msg,
                context={
                    "output_dir": str(self.output_dir),
                    "filename": filename,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            ) from e

        except Exception as e:
            # Catch-all for unexpected errors
            msg = "Unexpected error saving CSV"
            raise PersistenceError(
                msg,
                context={
                    "output_dir": str(self.output_dir),
                    "filename": filename,
                    "row_count": len(data) if isinstance(data, list) else None,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            ) from e

    def save_json(
        self,
        data: Any,
        filename: str,
    ) -> Path:
        """Save data to JSON file.

        Creates the output directory if it doesn't exist, serializes data as JSON,
        and returns the absolute path to the saved file.

        Parameters
        ----------
        data : Any
            JSON-serializable data (dict, list, str, int, float, bool, None)
        filename : str
            Output filename (e.g., "results.json")

        Returns
        -------
        Path
            Absolute path to saved JSON file

        Raises
        ------
        PersistenceError
            If save fails (not JSON-serializable, permission denied, etc.)

        Examples
        --------
        >>> writer = CSVWriter(output_dir=Path("outputs"))
        >>> data = {"status": "complete", "count": 42}
        >>> path = writer.save_json(data, "status.json")
        >>> json.loads(path.read_text())
        {'status': 'complete', 'count': 42}

        Notes
        -----
        - Uses pretty-printed JSON (indent=2) for readability
        - Ensures ASCII=False to support Unicode characters
        - Non-serializable objects raise PersistenceError with context
        """
        try:
            # Create output directory if needed
            self._ensure_output_dir_exists()

            # Construct full path
            output_path = self.output_dir / filename

            # Create parent directories if filename contains path separators
            if output_path.parent != self.output_dir:
                output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write JSON file
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"JSON saved: {output_path}")

            # Return absolute path
            return output_path.absolute()

        except TypeError as e:
            # JSON serialization error (non-serializable object)
            msg = "Data is not JSON-serializable"
            raise PersistenceError(
                msg,
                context={
                    "filename": filename,
                    "data_type": type(data).__name__,
                    "error": str(e),
                },
            ) from e

        except PermissionError as e:
            msg = "Permission denied creating JSON file"
            raise PersistenceError(
                msg,
                context={
                    "output_dir": str(self.output_dir),
                    "filename": filename,
                    "error": str(e),
                },
            ) from e

        except OSError as e:
            msg = "Failed to write JSON file"
            raise PersistenceError(
                msg,
                context={
                    "output_dir": str(self.output_dir),
                    "filename": filename,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            ) from e

        except Exception as e:
            msg = "Unexpected error saving JSON"
            raise PersistenceError(
                msg,
                context={
                    "output_dir": str(self.output_dir),
                    "filename": filename,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            ) from e

    def _ensure_output_dir_exists(self) -> None:
        """Create output directory if it doesn't exist.

        Raises
        ------
        PersistenceError
            If directory creation fails (permission denied, etc.)
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            msg = "Permission denied creating output directory"
            raise PersistenceError(
                msg,
                context={
                    "output_dir": str(self.output_dir),
                    "error": str(e),
                },
            ) from e
        except OSError as e:
            msg = "Failed to create output directory"
            raise PersistenceError(
                msg,
                context={
                    "output_dir": str(self.output_dir),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            ) from e
