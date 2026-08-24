"""Verification test asserting that no Python file in the project exceeds 350 lines of code."""
import os
import unittest


class TestLineCounts(unittest.TestCase):
    """Test suite ensuring strict compliance with codebase line count limits (< 350 lines)."""

    def test_all_python_files_under_350_lines(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        max_lines_allowed = 350
        offending_files = []

        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Skip virtual environments, external Carla-utils or third-party paper code if any
            if any(part in dirpath for part in [".git", "__pycache__", "venv", ".venv", "Carla-utils", "papers_and_code"]):
                continue

            for fname in filenames:
                if fname.endswith(".py"):
                    fpath = os.path.join(dirpath, fname)
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        line_count = len(f.readlines())

                    rel_path = os.path.relpath(fpath, root_dir)
                    if line_count > max_lines_allowed:
                        offending_files.append((rel_path, line_count))

        self.assertEqual(
            len(offending_files), 0,
            f"The following files exceed the strict {max_lines_allowed} lines limit: {offending_files}"
        )


if __name__ == "__main__":
    unittest.main()
