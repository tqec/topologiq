"""Utilities to assist the management of the graph manager BFS.

Usage:
    Call any function/class from a separate script.

"""

import shutil
from pathlib import Path


###################
# FILE MANAGEMENT #
###################
def rm_temp_files(temp_dir_path: Path):
    """Remove any temporary files created during run."""
    try:
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path)
    except (ValueError, FileNotFoundError) as e:
        print("Unable to delete temp files or temp folder does not exist", e)
