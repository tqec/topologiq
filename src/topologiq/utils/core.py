"""Utils shared by the core graph manager and pathfinder algorithms.

Usage:
    Call any function/class from a separate script.

"""

from datetime import datetime


def datetime_manager(
    t_1: datetime | None = None, t_2: datetime | None = None
) -> tuple[datetime, float]:
    """Start a timer or calculate times and durations.

    Args:
        t_1 (optional): A pre-existing start time, if available.
        t_2 (optional): A pre-existing end time, if available.

    Returns:
        t_1: A datestamp to be used as start time.
        duration: The duration between an incoming start and end times, or a calculated duration if no end time is provided.

    """
    t_1 = datetime.now() if not t_1 else t_1
    t_2 = datetime.now() if not t_2 else t_2
    duration = (t_2 - t_1).total_seconds()

    return t_1, duration
