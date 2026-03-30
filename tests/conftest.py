import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def tmp_graph_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
