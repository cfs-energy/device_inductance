from pathlib import Path
from mktestdocs import check_md_file


def test_md():
    """Test python snippets in markdown files"""
    check_md_file(Path(__file__).parent / "../README.md")
    check_md_file(Path(__file__).parent / "../docs/index.md")