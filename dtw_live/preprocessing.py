#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter as moving_average_filter

# from sklearn.base import TransformerMixin


def hampel_filter(series, size=10, n_sigmas=3, mode='reflect', return_index=False):
    """Hampel (median) filter implementation.

    Parameters
    ----------
    series : array-like
        Time series input.
    size : int or tuple of ints, optional
        The sizes of the filter for all axes, or for each axis separately.
    n_sigmas : float, optional
        Outlier thresholds expressed as the number of standard deviations.
    mode : str or tuple of str, optional
        Determines border overlap behaviour for the array.
            ‘reflect’ (d c b a | a b c d | d c b a)
                The input is extended by reflecting about the edge of the last
                pixel. This mode is also sometimes referred to as half-sample
                symmetric.
            ‘nearest’ (a a a a | a b c d | d d d d)
                The input is extended by replicating the last pixel.
            ‘mirror’ (d c b | a b c d | c b a)
                The input is extended by reflecting about the center of the
                last pixel. This mode is also sometimes referred to as
                whole-sample symmetric.
            ‘wrap’ (a b c d | a b c d | a b c d)
                The input is extended by wrapping around to the opposite edge.
    return_index : bool, optional
        Returns the index of detected outliers if True.

    Returns
    -------
    array-like
        Resultant filtered time series.
    array-like, optional
        Index of outliers.
    """
    new_series = series.copy()
    k = 1.4826  # median absolute deviation for normal distributions

    x = median_filter(series, size=size, mode=mode)
    S = k * median_filter(np.abs(series - x), size=size, mode=mode)
    mask = np.abs(new_series - x) > n_sigmas * S
    np.putmask(new_series, mask, x)

    if return_index:
        return new_series, np.argwhere(mask)
    else:
        return new_series
