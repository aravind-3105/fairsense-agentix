"""Prompt template loader with variable substitution.

This module provides the PromptLoader class for loading versioned prompt templates
from text files and performing variable substitution using Python's string.Template.

Template files should use $variable or ${variable} syntax for substitution:
    - $variable: Simple variable (e.g., $text, $image_path)
    - ${variable}: Bracketed variable for complex names (e.g., ${user_input})
    - $$: Escaped dollar sign (literal $)

Examples
--------
    >>> loader = PromptLoader()
    >>> prompt = loader.load("bias_text_v1", text="Job posting content")
    >>> print(prompt)
    You are an AI bias detection expert...

    # Custom template directory
    >>> loader = PromptLoader(templates_dir=Path("custom/templates"))
    >>> prompt = loader.load("custom_prompt_v1", var1="value1")
"""

from pathlib import Path
from string import Template


class PromptLoader:
    """Load and render prompt templates with variable substitution.

    Prompt templates are stored as .txt files in the templates directory.
    Each template is versioned (e.g., bias_text_v1.txt, bias_text_v2.txt)
    to support A/B testing and reproducibility.

    Parameters
    ----------
    templates_dir : Path | None, optional
        Directory containing template files, by default None (uses bundled templates)

    Attributes
    ----------
    templates_dir : Path
        Path to directory containing .txt template files

    Raises
    ------
    FileNotFoundError
        If templates directory doesn't exist
    """

    def __init__(self, templates_dir: Path | None = None) -> None:
        """Initialize PromptLoader with template directory.

        Parameters
        ----------
        templates_dir : Path | None, optional
            Directory containing template files. If None, uses the default
            templates directory bundled with the package, by default None
        """
        if templates_dir is None:
            # Default to templates/ subdirectory next to this file
            templates_dir = Path(__file__).parent / "templates"

        self.templates_dir = templates_dir

        if not self.templates_dir.exists():
            msg = f"Templates directory not found: {self.templates_dir}"
            raise FileNotFoundError(msg)

    def load(self, template_name: str, **variables: str | int | float | bool) -> str:
        """Load template and substitute variables.

        Parameters
        ----------
        template_name : str
            Name of template file without .txt extension
            (e.g., "bias_text_v1" for "bias_text_v1.txt")
        **variables : str | int | float | bool
            Variables to substitute in template. Keys should match
            $variable names in the template file.

        Returns
        -------
        str
            Rendered prompt with variables substituted

        Raises
        ------
        FileNotFoundError
            If template file doesn't exist
        ValueError
            If template has required variables that weren't provided

        Examples
        --------
        >>> loader = PromptLoader()
        >>> prompt = loader.load("bias_text_v1", text="Sample text")
        >>> prompt = loader.load("bias_image_v1", caption="A person", ocr="Job posting")
        """
        normalized_name = template_name.removesuffix(".txt")
        template_path = self.templates_dir / f"{normalized_name}.txt"

        if not template_path.exists():
            msg = (
                f"Template not found: {normalized_name}.txt in {self.templates_dir}. "
                f"Available templates: {self.list_templates()}"
            )
            raise FileNotFoundError(msg)

        template_text = template_path.read_text(encoding="utf-8")
        template = Template(template_text)

        # Use safe_substitute to allow missing variables (they remain as $var)
        # If strict substitution is needed, use template.substitute() instead
        try:
            rendered = template.substitute(**variables)
        except KeyError as e:
            msg = (
                f"Missing required variable in template {template_name}: {e}. "
                f"Provided variables: {list(variables.keys())}"
            )
            raise ValueError(msg) from e

        return rendered

    def list_templates(self) -> list[str]:
        """List all available template names.

        Returns
        -------
        list[str]
            List of template names (without .txt extension)

        Examples
        --------
        >>> loader = PromptLoader()
        >>> templates = loader.list_templates()
        >>> print(templates)
        ['bias_text_v1', 'bias_image_v1']
        """
        if not self.templates_dir.exists():
            return []

        template_files = self.templates_dir.glob("*.txt")
        return sorted(path.stem for path in template_files)

    def load_raw(self, template_name: str) -> str:
        """Load template without variable substitution.

        Useful for inspecting template structure or debugging.

        Parameters
        ----------
        template_name : str
            Name of template file without .txt extension

        Returns
        -------
        str
            Raw template text with $variables intact

        Raises
        ------
        FileNotFoundError
            If template file doesn't exist

        Examples
        --------
        >>> loader = PromptLoader()
        >>> raw = loader.load_raw("bias_text_v1")
        >>> print(raw)
        You are an AI bias detection expert. Analyze: $text
        """
        normalized_name = template_name.removesuffix(".txt")
        template_path = self.templates_dir / f"{normalized_name}.txt"

        if not template_path.exists():
            msg = f"Template not found: {normalized_name}.txt in {self.templates_dir}"
            raise FileNotFoundError(msg)

        return template_path.read_text(encoding="utf-8")
