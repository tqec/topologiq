"""Pytest configuration and fixtures for topologiq tests."""

import pytest


@pytest.fixture
def simple_graph():
    """Fixture providing a simple test graph.

    Returns a basic graph structure compatible with topologiq's
    SimpleDictGraph format.
    """
    return {
        "nodes": [
            {"id": 0, "type": "X"},
            {"id": 1, "type": "Z"},
        ],
        "edges": [
            {"source": 0, "target": 1, "type": "plain"},
        ],
    }
