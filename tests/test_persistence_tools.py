"""Tests for persistence tools.

This module tests the persistence tool implementations, including protocol
compliance, fake tool behavior, registry resolution, and real tool functionality.
"""

import csv
import json
from pathlib import Path

import pytest

from fairsense_agentix.tools.exceptions import PersistenceError
from fairsense_agentix.tools.fake import FakePersistenceTool
from fairsense_agentix.tools.interfaces import PersistenceTool
from fairsense_agentix.tools.persistence import CSVWriter


pytestmark = pytest.mark.unit


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestProtocolCompliance:
    """Test that persistence implementations satisfy the protocol."""

    def test_fake_persistence_satisfies_protocol(self):
        """Verify FakePersistenceTool implements PersistenceTool protocol."""
        fake_persistence = FakePersistenceTool(output_dir=Path("outputs"))
        assert isinstance(fake_persistence, PersistenceTool)

    def test_csv_writer_satisfies_protocol(self):
        """Verify CSVWriter implements PersistenceTool protocol."""
        writer = CSVWriter(output_dir=Path("outputs"))
        assert isinstance(writer, PersistenceTool)


# ============================================================================
# Fake Persistence Tests
# ============================================================================


class TestFakePersistence:
    """Test fake persistence for unit testing and development."""

    def test_fake_returns_paths_without_writing(self, tmp_path):
        """Test fake persistence returns paths but doesn't write files."""
        fake = FakePersistenceTool(
            output_dir=tmp_path / "outputs", actually_write=False
        )

        data = [{"col1": "value1"}]
        path = fake.save_csv(data, "test.csv")

        # Should return path
        assert isinstance(path, Path)
        assert path.name == "test.csv"

        # But file should NOT exist (actually_write=False)
        assert not path.exists()

    def test_fake_writes_when_configured(self, tmp_path):
        """Test fake persistence can actually write files when enabled."""
        fake = FakePersistenceTool(output_dir=tmp_path, actually_write=True)

        data = [{"col1": "value1", "col2": "value2"}]
        path = fake.save_csv(data, "test.csv")

        # File should exist
        assert path.exists()

        # Verify content
        content = path.read_text()
        assert "col1" in content
        assert "value1" in content

    def test_fake_tracks_calls(self, tmp_path):
        """Test fake persistence tracks method calls."""
        fake = FakePersistenceTool(output_dir=tmp_path)

        assert fake.call_count_csv == 0

        fake.save_csv([{"a": 1}], "file1.csv")
        assert fake.call_count_csv == 1

        fake.save_csv([{"b": 2}], "file2.csv")
        assert fake.call_count_csv == 2

    def test_fake_records_saved_files(self, tmp_path):
        """Test fake persistence records list of saved files."""
        fake = FakePersistenceTool(output_dir=tmp_path)

        assert len(fake.saved_files) == 0

        path1 = fake.save_csv([{"a": 1}], "file1.csv")
        assert path1 in fake.saved_files

        path2 = fake.save_json({"key": "value"}, "file2.json")
        assert path2 in fake.saved_files

        assert len(fake.saved_files) == 2


# ============================================================================
# CSVWriter - CSV Tests
# ============================================================================


class TestCSVWriterCSV:
    """Test CSV writer CSV functionality."""

    def test_save_csv_basic(self, tmp_path):
        """Test basic CSV saving."""
        writer = CSVWriter(output_dir=tmp_path)

        data = [
            {"name": "Alice", "age": 30, "score": 0.95},
            {"name": "Bob", "age": 25, "score": 0.87},
        ]

        path = writer.save_csv(data, "test.csv")

        # Verify path
        assert path.exists()
        assert path.is_absolute()
        assert path.name == "test.csv"
        assert path.parent == tmp_path

        # Verify content
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"
        assert rows[0]["age"] == "30"
        assert rows[1]["name"] == "Bob"

    def test_save_csv_creates_directory(self, tmp_path):
        """Test CSV writer creates output directory if needed."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        writer = CSVWriter(output_dir=nested_dir)

        data = [{"col": "value"}]
        path = writer.save_csv(data, "test.csv")

        # Directory should be created
        assert nested_dir.exists()
        assert path.exists()

    def test_save_csv_empty_data(self, tmp_path):
        """Test saving empty data list."""
        writer = CSVWriter(output_dir=tmp_path)

        path = writer.save_csv(data=[], filename="empty.csv")

        # File should exist
        assert path.exists()

        # File should be empty or have just headers
        content = path.read_text()
        assert len(content) == 0 or content.strip() == ""

    def test_save_csv_missing_values(self, tmp_path):
        """Test CSV handles missing values in rows."""
        writer = CSVWriter(output_dir=tmp_path)

        data = [
            {"col1": "a", "col2": "b", "col3": "c"},
            {"col1": "d", "col3": "f"},  # Missing col2
        ]

        path = writer.save_csv(data, "missing.csv")

        # Read back
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Second row should have empty string for col2
        assert rows[1]["col1"] == "d"
        assert rows[1]["col2"] == ""
        assert rows[1]["col3"] == "f"

    def test_save_csv_special_characters(self, tmp_path):
        """Test CSV handles special characters."""
        writer = CSVWriter(output_dir=tmp_path)

        data = [
            {"text": 'Quote: "hello"', "comma": "a,b,c"},
            {"text": "Newline:\nNext line", "comma": "x,y,z"},
        ]

        path = writer.save_csv(data, "special.csv")

        # Read back
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # CSV should properly escape quotes and commas
        assert rows[0]["text"] == 'Quote: "hello"'
        assert rows[0]["comma"] == "a,b,c"

    def test_save_csv_unicode(self, tmp_path):
        """Test CSV handles Unicode characters."""
        writer = CSVWriter(output_dir=tmp_path)

        data = [
            {"text": "Hello 世界", "emoji": "🎉 ✨"},
            {"text": "Café", "emoji": "☕"},
        ]

        path = writer.save_csv(data, "unicode.csv")

        # Read back with UTF-8 encoding
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["text"] == "Hello 世界"
        assert rows[0]["emoji"] == "🎉 ✨"

    def test_save_csv_returns_absolute_path(self, tmp_path):
        """Test CSV writer always returns absolute paths."""
        # Use relative path for output_dir
        relative_dir = Path("outputs")
        writer = CSVWriter(output_dir=relative_dir)

        # Create directory for the test
        abs_dir = (tmp_path / "outputs").resolve()
        abs_dir.mkdir(parents=True, exist_ok=True)

        # Change to tmp_path so relative path resolves there
        import os  # noqa: PLC0415

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            data = [{"col": "value"}]
            path = writer.save_csv(data, "test.csv")

            # Path must be absolute
            assert path.is_absolute()

        finally:
            os.chdir(old_cwd)


# ============================================================================
# CSVWriter - JSON Tests
# ============================================================================


class TestCSVWriterJSON:
    """Test CSV writer JSON functionality."""

    def test_save_json_basic(self, tmp_path):
        """Test basic JSON saving."""
        writer = CSVWriter(output_dir=tmp_path)

        data = {"status": "complete", "count": 42, "items": ["a", "b", "c"]}

        path = writer.save_json(data, "test.json")

        # Verify path
        assert path.exists()
        assert path.is_absolute()
        assert path.name == "test.json"

        # Verify content
        loaded = json.loads(path.read_text())
        assert loaded["status"] == "complete"
        assert loaded["count"] == 42
        assert loaded["items"] == ["a", "b", "c"]

    def test_save_json_creates_directory(self, tmp_path):
        """Test JSON writer creates output directory if needed."""
        nested_dir = tmp_path / "json" / "outputs"
        writer = CSVWriter(output_dir=nested_dir)

        data = {"key": "value"}
        path = writer.save_json(data, "test.json")

        # Directory should be created
        assert nested_dir.exists()
        assert path.exists()

    def test_save_json_pretty_printed(self, tmp_path):
        """Test JSON is pretty-printed (indented)."""
        writer = CSVWriter(output_dir=tmp_path)

        data = {"key1": "value1", "key2": "value2"}
        path = writer.save_json(data, "pretty.json")

        content = path.read_text()

        # Should be indented (pretty-printed)
        assert "\n" in content
        assert "  " in content  # Indentation spaces

    def test_save_json_unicode(self, tmp_path):
        """Test JSON handles Unicode characters."""
        writer = CSVWriter(output_dir=tmp_path)

        data = {"text": "Hello 世界", "emoji": "🎉"}
        path = writer.save_json(data, "unicode.json")

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["text"] == "Hello 世界"
        assert loaded["emoji"] == "🎉"

    def test_save_json_non_serializable_raises_error(self, tmp_path):
        """Test non-JSON-serializable data raises PersistenceError."""
        writer = CSVWriter(output_dir=tmp_path)

        # Create non-serializable object
        class NotSerializable:
            pass

        data = {"object": NotSerializable()}

        with pytest.raises(PersistenceError) as exc_info:
            writer.save_json(data, "bad.json")

        assert "serializable" in str(exc_info.value).lower()
        assert "data_type" in exc_info.value.context

    def test_save_json_returns_absolute_path(self, tmp_path):
        """Test JSON writer always returns absolute paths."""
        writer = CSVWriter(output_dir=Path("relative_outputs"))

        # Adjust for test
        abs_dir = (tmp_path / "relative_outputs").resolve()
        abs_dir.mkdir(parents=True, exist_ok=True)

        import os  # noqa: PLC0415

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            data = {"key": "value"}
            path = writer.save_json(data, "test.json")

            # Path must be absolute
            assert path.is_absolute()

        finally:
            os.chdir(old_cwd)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_csv_error_includes_context(self, tmp_path):
        """Test CSV errors include helpful context."""
        writer = CSVWriter(output_dir=tmp_path)

        # Create data that will cause an error during CSV writing
        class BadData:
            def __iter__(self):
                raise ValueError("Cannot iterate")

        # Try to save invalid data
        with pytest.raises(PersistenceError) as exc_info:
            writer.save_csv(data=BadData(), filename="bad.csv")  # type: ignore

        # Error should have context
        error = exc_info.value
        assert error.context is not None
        assert "filename" in error.context

    def test_json_error_includes_context(self, tmp_path):
        """Test JSON errors include helpful context."""
        writer = CSVWriter(output_dir=tmp_path)

        # Non-serializable data
        class Bad:
            pass

        with pytest.raises(PersistenceError) as exc_info:
            writer.save_json(data={"obj": Bad()}, filename="bad.json")

        # Error should have context
        error = exc_info.value
        assert error.context is not None
        assert "filename" in error.context

    def test_permission_denied_handled(self, tmp_path):
        """Test permission denied errors are caught and wrapped."""
        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only

        writer = CSVWriter(output_dir=readonly_dir / "subdir")

        try:
            with pytest.raises(PersistenceError) as exc_info:
                writer.save_csv([{"col": "value"}], "test.csv")

            assert "permission" in str(exc_info.value).lower()
        finally:
            # Cleanup: restore permissions
            readonly_dir.chmod(0o755)


# ============================================================================
# Path Tests
# ============================================================================


class TestPathHandling:
    """Test path resolution and handling."""

    def test_output_dir_resolved_to_absolute(self, tmp_path):
        """Test output_dir is resolved to absolute path."""
        writer = CSVWriter(output_dir=Path("relative"))

        # output_dir should be absolute
        assert writer.output_dir.is_absolute()

    def test_nested_filename_not_allowed(self, tmp_path):
        """Test that nested filenames (with /) are handled correctly."""
        writer = CSVWriter(output_dir=tmp_path)

        # Filename with path separator should just be treated as filename
        # (no directory traversal)
        data = [{"col": "value"}]

        # This should work - the / is part of the filename
        path = writer.save_csv(data, "subdir/test.csv")

        # The subdir should have been created
        assert path.exists()
        assert path.parent.name == "subdir"
