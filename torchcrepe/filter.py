import numpy as np
import torch


###############################################################################
# Sequence filters
###############################################################################


def mean(signals, win_length=9):
    """Averave filtering for signals containing nan values

    Arguments
        signals (torch.tensor (shape=(batch, time)))
            The signals to filter
        win_length
            The size of the analysis window

    Returns
        filtered (torch.tensor (shape=(batch, time)))
    """
    return nanfilter(signals, win_length, nanmean)


def median(signals, win_length):
    """Median filtering for signals containing nan values

    Arguments
        signals (torch.tensor (shape=(batch, time)))
            The signals to filter
        win_length
            The size of the analysis window

    Returns
        filtered (torch.tensor (shape=(batch, time)))
    """
    return nanfilter(signals, win_length, nanmedian)


###############################################################################
# Utilities
###############################################################################


def nanfilter(signals, win_length, filter_fn):
    """Filters a sequence, ignoring nan values

    Arguments
        signals (torch.tensor (shape=(batch, time)))
            The signals to filter
        win_length
            The size of the analysis window
        filter_fn (function)
            The function to use for filtering

    Returns
        filtered (torch.tensor (shape=(batch, time)))
    """
    # Output buffer
    filtered = torch.empty_like(signals)

    # Loop over frames
    for i in range(signals.size(1)):

        # Get analysis window bounds
        start = max(0, i - win_length // 2)
        end = min(signals.size(1), i + win_length // 2 + 1)

        # Apply filter to window
        filtered[:, i] = filter_fn(signals[:, start:end])

    return filtered


def nanmean(signals):
    """Computes the mean, ignoring nans

    Arguments
        signals (torch.tensor [shape=(batch, time)])
            The signals to filter

    Returns
        filtered (torch.tensor [shape=(batch, time)])
    """
    signals = signals.clone()

    # Find nans
    nans = torch.isnan(signals)

    # Set nans to 0.
    signals[nans] = 0.

    # Compute average
    return signals.sum(dim=1) / (~nans).float().sum(dim=1)


def nanmedian(signals):
    """Computes the median, ignoring nans

    Arguments
        signals (torch.tensor [shape=(batch, time)])
            The signals to filter

    Returns
        filtered (torch.tensor [shape=(batch, time)])
    """
    # Find nans
    nans = torch.isnan(signals)

    # Compute median for each slice
    medians = [nanmedian1d(signal[~nan]) for signal, nan in zip(signals, nans)]

    # Stack results
    return torch.tensor(medians, dtype=signals.dtype, device=signals.device)


def nanmedian1d(signal):
    """Computes the median. If signal is empty, returns torch.nan

    Arguments
        signal (torch.tensor [shape=(time,)])

    Returns
        median (torch.tensor [shape=(1,)])
    """
    return torch.median(signal) if signal.numel() else np.nan
