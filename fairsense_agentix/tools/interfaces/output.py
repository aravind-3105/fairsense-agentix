"""Formatting and persistence tool protocol interfaces."""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FormatterTool(Protocol):
    """Protocol for output formatting tools.

    Generates formatted output (HTML with highlights, tables) from structured data.

    Examples
    --------
    >>> formatter = HTMLFormatter()
    >>> html = formatter.highlight(
    ...     text="Job posting text",
    ...     spans=[(0, 3, "gender")],
    ...     bias_types={"gender": "#FFB3BA"},
    ... )
    >>> table = formatter.table(
    ...     data=[{"risk": "Bias", "severity": "High"}], headers=["Risk", "Severity"]
    ... )
    """

    def highlight(
        self,
        text: str,
        spans: list[tuple[int, int, str]],
        bias_types: dict[str, str],
    ) -> str:
        """Generate HTML with highlighted text spans.

        Parameters
        ----------
        text : str
            Text to highlight
        spans : list[tuple[int, int, str]]
            List of (start_idx, end_idx, bias_type) tuples
        bias_types : dict[str, str]
            Mapping of bias_type to color (e.g., {"gender": "#FFB3BA"})

        Returns
        -------
        str
            HTML string with highlighted spans

        Raises
        ------
        FormatterError
            If formatting fails (invalid spans, template error, etc.)
        """
        ...

    def highlight_fragment(
        self,
        text: str,
        spans: list[tuple[int, int, str]],
        bias_types: dict[str, str],
    ) -> str:
        """Generate HTML fragment (not complete document) for embedding.

        Parameters
        ----------
        text : str
            Text to highlight
        spans : list[tuple[int, int, str]]
            List of (start_idx, end_idx, bias_type) tuples
        bias_types : dict[str, str]
            Mapping of bias_type to color

        Returns
        -------
        str
            HTML fragment with highlighted spans

        Raises
        ------
        FormatterError
            If formatting fails
        """
        ...

    def table(
        self,
        data: list[dict[str, Any]],
        headers: list[str] | None = None,
    ) -> str:
        """Generate HTML table from structured data.

        Parameters
        ----------
        data : list[dict[str, Any]]
            List of row dictionaries
        headers : list[str] | None, optional
            Column headers (if None, infer from first row keys), by default None

        Returns
        -------
        str
            HTML table string

        Raises
        ------
        FormatterError
            If formatting fails (empty data, invalid structure, etc.)
        """
        ...


@runtime_checkable
class PersistenceTool(Protocol):
    """Protocol for file persistence tools.

    Saves data to disk in various formats (CSV, JSON, etc.).

    Examples
    --------
    >>> persistence = FileWriter(output_dir="outputs")
    >>> csv_path = persistence.save_csv(
    ...     data=[{"risk": "Bias", "severity": "High"}], filename="risks.csv"
    ... )
    >>> print(csv_path)
    /absolute/path/to/outputs/risks.csv
    """

    def save_csv(
        self,
        data: list[dict[str, Any]],
        filename: str,
    ) -> Path:
        """Save data to CSV file.

        Parameters
        ----------
        data : list[dict[str, Any]]
            List of row dictionaries
        filename : str
            Output filename (e.g., "risks.csv")

        Returns
        -------
        Path
            Absolute path to saved CSV file

        Raises
        ------
        PersistenceError
            If save fails (permission denied, disk full, etc.)
        """
        ...

    def save_json(
        self,
        data: Any,
        filename: str,
    ) -> Path:
        """Save data to JSON file.

        Parameters
        ----------
        data : Any
            JSON-serializable data
        filename : str
            Output filename (e.g., "result.json")

        Returns
        -------
        Path
            Absolute path to saved JSON file

        Raises
        ------
        PersistenceError
            If save fails or data not JSON-serializable
        """
        ...
