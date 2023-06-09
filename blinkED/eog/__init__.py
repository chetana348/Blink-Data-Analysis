"""Submodule for NeuroKit."""

from ..signal import signal_rate as eog_rate
from .eog_analyze import eog_analyze
from .eog_clean import eog_clean
from .eog_eventrelated import eog_eventrelated
from .eog_features import eog_features
from .eog_findpeaks import eog_findpeaks
from .eog_intervalrelated import eog_intervalrelated
from .eog_peaks import eog_peaks
from .eog_plot import eog_plot
from .eog_process import eog_process
from .eog_read import eog_read
from .eog_read1 import eog_read1

__all__ = [
    "eog_rate",
    "eog_clean",
    "eog_features",
    "eog_findpeaks",
    "eog_peaks",
    "eog_process",
    "eog_plot",
    "eog_eventrelated",
    "eog_intervalrelated",
    "eog_analyze",
    "eog_read",
    "eog_read1"
]
