"""Read and/or auto-complete incoming KWARGs.

KWARGs can be passed by `check_assemble_kwargs` to ensure all KWARGs are present
and auto-complete any KWARGs missing in call. When Topologiq is called without KWARGs,
`check_assemble_kwargs` can be used to create a full set of KWARGs. In practice, this means
the user should only specifically give KWARGs when he/she desires to deviate from the
default KWARGs values given in `./kwargs.py`

Usage:
    Call `check_assemble_kwargs()` programmatically from a separate script.

"""

from topologiq.kwargs import (
    BEAMS_SHORT_LEN,
    DEBUG,
    FIRST_ID_STRATEGY,
    HIDE_PORTS,
    LOG_STATS,
    MAX_ATTEMPTS,
    MIN_SUCC_RATE,
    SEED,
    STOP_ON_FIRST_SUCCESS,
    STRIP_PORTS,
    VALUE_FUNCTION_HYPERPARAMS,
)
from topologiq.utils.core import datetime_manager


##################
# KWARGs MANAGER #
##################
def check_assemble_kwargs(**kwargs) -> dict[str, any]:
    """Check if all kwargs are present and add any missing.

    Args:
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, this function will auto-complete it based on
            on `./src/topologiq/kwargs.py`.

    """

    if len(kwargs) == 0:
        kwargs = {
            "weights": VALUE_FUNCTION_HYPERPARAMS,
            "first_id_strategy": FIRST_ID_STRATEGY,
            "beams_len_short": BEAMS_SHORT_LEN,
            "seed": SEED,
            "vis_options": (None, None),
            "max_attempts": MAX_ATTEMPTS,
            "stop_on_first_success": STOP_ON_FIRST_SUCCESS,
            "min_succ_rate": MIN_SUCC_RATE,
            "strip_ports": STRIP_PORTS,
            "hide_ports": HIDE_PORTS,
            "log_stats": LOG_STATS,
            "log_stats_id": None,
            "debug": DEBUG,
        }

    if "weights" not in kwargs:
        kwargs["weights"] = VALUE_FUNCTION_HYPERPARAMS
    if "first_id_strategy" not in kwargs:
        kwargs["first_id_strategy"] = FIRST_ID_STRATEGY
    if "beams_len_short" not in kwargs:
        kwargs["beams_len_short"] = BEAMS_SHORT_LEN
    if "seed" not in kwargs:
        kwargs["seed"] = SEED
    if "vis_options" not in kwargs:
        kwargs["vis_options"] = (None, None)
    if "max_attempts" not in kwargs:
        kwargs["max_attempts"] = MAX_ATTEMPTS
    if "stop_on_first_success" not in kwargs:
        kwargs["stop_on_first_success"] = STOP_ON_FIRST_SUCCESS
    if "min_succ_rate" not in kwargs:
        kwargs["min_succ_rate"] = MIN_SUCC_RATE
    if "strip_ports" not in kwargs:
        kwargs["strip_ports"] = STRIP_PORTS
    if "hide_ports" not in kwargs:
        kwargs["hide_ports"] = HIDE_PORTS
    if "log_stats" not in kwargs:
        kwargs["log_stats"] = LOG_STATS
    if "log_stats_id" not in kwargs:
        kwargs["log_stats_id"] = None
    if "debug" not in kwargs:
        kwargs["debug"] = DEBUG

    # Create unique run ID if stats logging is on
    if kwargs["log_stats"]:
        timestamp, _ = datetime_manager()
        kwargs["log_stats_id"] = timestamp.strftime("%Y%m%d_%H%M%S_%f")

    return kwargs
