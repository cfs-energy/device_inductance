"""Run each example file, failing the test on any errors"""

import os
import pathlib
import runpy

import pytest

# os.environ["DEVICE_INDUCTANCE_TESTING"] = "True"

EXAMPLES_DIR = pathlib.Path(__file__).parent / "../examples"
EXAMPLES = [
    EXAMPLES_DIR / x
    for x in os.listdir(EXAMPLES_DIR)
    if (os.path.isfile(EXAMPLES_DIR / x) and x[-3:] == ".py")
]


@pytest.mark.parametrize("example_file", EXAMPLES)
def test_example(example_file: pathlib.Path):
    """Test every example file"""
    runpy.run_path(str(example_file), run_name="__main__")
