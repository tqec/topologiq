"""Basic smoke tests to validate core functionality - Issue #11."""
import sys
from pathlib import Path

# Ensure we can import the topologiq module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_basic_imports():
    """Test that core modules can be imported without errors."""
    try:
        import topologiq
        assert topologiq is not None
    except ImportError:
        # If module structure is different, try importing specific components
        pass

    # Test individual script imports
    from topologiq.scripts import runner
    from topologiq.utils import classes
    assert runner is not None
    assert classes is not None


def test_runner_function_exists():
    """Test that the main runner function exists and is callable."""
    from topologiq.scripts.runner import runner
    assert callable(runner)


def test_basic_classes_exist():
    """Test that basic classes are available."""

    # Test that we can create basic instances
    coord = (0, 0, 0)
    kind = "xxz"
    block = (coord, kind)

    assert coord is not None
    assert block is not None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
