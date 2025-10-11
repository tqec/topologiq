"""Tests for topologiq.utils.utils_misc module."""

import numpy as np

from topologiq.utils import utils_misc


class TestManhattanDistance:
    """Tests for Manhattan distance calculations."""

    def test_get_manhattan_basic(self):
        """Test Manhattan distance between two 3D points."""
        src = (0, 0, 0)
        tgt = (1, 1, 1)
        distance = utils_misc.get_manhattan(src, tgt)
        assert distance == 3

    def test_get_manhattan_same_point(self):
        """Test Manhattan distance from a point to itself."""
        coord = (5, 3, 2)
        distance = utils_misc.get_manhattan(coord, coord)
        assert distance == 0

    def test_get_manhattan_negative_coords(self):
        """Test Manhattan distance with negative coordinates."""
        src = (-1, -2, -3)
        tgt = (1, 2, 3)
        distance = utils_misc.get_manhattan(src, tgt)
        assert distance == 12

    def test_get_manhattan_2d_coords(self):
        """Test Manhattan distance works with 2D coordinates."""
        src = (0, 0)
        tgt = (3, 4)
        distance = utils_misc.get_manhattan(src, tgt)
        assert distance == 7

    def test_get_manhattan_large_coords(self):
        """Test Manhattan distance with large coordinate values."""
        src = (0, 0, 0)
        tgt = (100, 200, 300)
        distance = utils_misc.get_manhattan(src, tgt)
        assert distance == 600


class TestMaxManhattan:
    """Tests for maximum Manhattan distance calculations."""

    def test_get_max_manhattan_basic(self):
        """Test max Manhattan distance from a point to a list of points."""
        src = (0, 0, 0)
        coords = [(1, 0, 0), (0, 2, 0), (0, 0, 5)]
        max_dist = utils_misc.get_max_manhattan(src, coords)
        assert max_dist == 5

    def test_get_max_manhattan_empty_list(self):
        """Test max Manhattan distance with empty coordinate list."""
        src = (0, 0, 0)
        coords = []
        max_dist = utils_misc.get_max_manhattan(src, coords)
        assert max_dist == 0

    def test_get_max_manhattan_single_point(self):
        """Test max Manhattan distance with single coordinate."""
        src = (0, 0, 0)
        coords = [(3, 4, 0)]
        max_dist = utils_misc.get_max_manhattan(src, coords)
        assert max_dist == 7

    def test_get_max_manhattan_includes_source(self):
        """Test max Manhattan distance when list includes source point."""
        src = (1, 1, 1)
        coords = [(1, 1, 1), (2, 2, 2), (5, 5, 5)]
        max_dist = utils_misc.get_max_manhattan(src, coords)
        assert max_dist == 12

    def test_get_max_manhattan_negative_coords(self):
        """Test max Manhattan distance with negative coordinates."""
        src = (0, 0, 0)
        coords = [(-5, 0, 0), (5, 0, 0), (0, -10, 0)]
        max_dist = utils_misc.get_max_manhattan(src, coords)
        assert max_dist == 10


class TestConstants:
    """Tests for module-level constants."""

    def test_header_bfs_manager_stats_exists(self):
        """Test that BFS manager stats header is defined."""
        assert hasattr(utils_misc, "HEADER_BFS_MANAGER_STATS")
        assert isinstance(utils_misc.HEADER_BFS_MANAGER_STATS, list)
        assert len(utils_misc.HEADER_BFS_MANAGER_STATS) > 0

    def test_header_pathfinder_stats_exists(self):
        """Test that pathfinder stats header is defined."""
        assert hasattr(utils_misc, "HEADER_PATHFINDER_STATS")
        assert isinstance(utils_misc.HEADER_PATHFINDER_STATS, list)
        assert len(utils_misc.HEADER_PATHFINDER_STATS) > 0

    def test_header_output_stats_exists(self):
        """Test that output stats header is defined."""
        assert hasattr(utils_misc, "HEADER_OUTPUT_STATS")
        assert isinstance(utils_misc.HEADER_OUTPUT_STATS, list)
        assert len(utils_misc.HEADER_OUTPUT_STATS) > 0

    def test_headers_contain_expected_fields(self):
        """Test that headers contain expected field names."""
        # BFS manager should track unique run ID
        assert "unique_run_id" in utils_misc.HEADER_BFS_MANAGER_STATS
        assert "run_success" in utils_misc.HEADER_BFS_MANAGER_STATS

        # Pathfinder should track iteration info
        assert "unique_run_id" in utils_misc.HEADER_PATHFINDER_STATS
        assert "iter_success" in utils_misc.HEADER_PATHFINDER_STATS

        # Output stats should track basic info
        assert "unique_run_id" in utils_misc.HEADER_OUTPUT_STATS
        assert "circuit_name" in utils_misc.HEADER_OUTPUT_STATS


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_manhattan_with_floats(self):
        """Test Manhattan distance with floating point coordinates."""
        src = (0.0, 0.0, 0.0)
        tgt = (1.5, 2.5, 3.5)
        distance = utils_misc.get_manhattan(src, tgt)
        assert np.isclose(distance, 7.5)

    def test_max_manhattan_all_equidistant(self):
        """Test max Manhattan distance when all points are equidistant."""
        src = (0, 0, 0)
        coords = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        max_dist = utils_misc.get_max_manhattan(src, coords)
        assert max_dist == 1

    def test_manhattan_high_dimension(self):
        """Test Manhattan distance works with higher dimensional coords."""
        src = (0, 0, 0, 0)
        tgt = (1, 1, 1, 1)
        distance = utils_misc.get_manhattan(src, tgt)
        assert distance == 4
