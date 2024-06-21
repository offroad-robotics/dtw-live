#!/usr/bin/env python3

from math import erf
from warnings import warn

import numpy as np


def to_padded_ndarray(arr, pad_value=np.nan, dtype=np.float64):
    """Convert an array to a 3D array padded with NaN values.

    Parameters
    ----------
    array : array-like
        Array to convert.
    dtype : data type, default=numpy.float
        Data type for the returned array.

    Returns
    -------
    array-like with shape (n_samples, n_timesteps, n_features)
    """
    if arr[0].ndim == 2:
        if np.isnan(pad_value):
            arr = [a[~np.isnan(a).all(axis=-1)] for a in arr]
        else:
            arr = [a[(a != pad_value).all(axis=-1)] for a in arr]

        max_size = max(a.shape[0] for a in arr)
        a0 = arr[0].shape[1]
        if not all(a.shape[1] == a0 for a in arr):
            raise ValueError('Final 3D array axis must have equal dimensions')

        pad_shape = (len(arr), max_size, a0)
    else:
        if np.isnan(pad_value):
            arr = [a[~np.isnan(a)] for a in arr]
        else:
            arr = [a[a != pad_value] for a in arr]

        max_size = max(a.shape[0] for a in arr)
        pad_shape = (len(arr), max_size)

    if np.isnan(pad_value) and dtype != np.float64:
        raise ValueError('Nan cannot be used for non-floating point types')

    pad_array = np.full(pad_shape, pad_value, dtype=dtype)
    for i, a in enumerate(arr):
        pad_array[i, :a.shape[0]] = a

    return pad_array


def to_ragged_array(arr, pad_value=np.nan, dtype=np.float64):
    """Convert a NaN-padded ndarray (from :meth:`to_padded_ndarray` format) to
    a list of ndarrays without padding.

    Parameters
    ----------
    array : ndarray with shape (n_samples, n_timesteps, n_features).
        Padded array to convert.
    dtype : data type, default=numpy.float
        Data type for the returned array.

    Returns
    -------
    list of ndarrays with shape (n_samples,)
        Ragged array (list of ndarrays)
    """
    if isinstance(arr, list):
        if all(isinstance(a, np.ndarray) for a in arr):
            if np.isnan(pad_value):
                return [a[~np.isnan(a).all(axis=-1)] for a in arr]
            else:
                return [a[(a != pad_value).all(axis=-1)] for a in arr]
        else:
            raise ValueError('Invalid data format (must be ndarray or list\
                              of ndarrays).')

    if arr.ndim == 3:
        rarr = []
        for i, s in enumerate(arr):
            if np.isnan(pad_value):
                sr = np.array(s[~np.isnan(s).all(axis=-1)], dtype=dtype)
            else:
                sr = np.array(s[(s != pad_value).all(axis=-1)], dtype=dtype)

            if not sr.flags['C_CONTIGUOUS']:
                sr = np.ascontiguousarray(sr)

            rarr.append(sr)

    elif arr.ndim == 2:
        if np.isnan(pad_value):
            rarr = [np.array(arr[~np.isnan(arr).all(axis=-1)], dtype=dtype)]
        else:
            rarr = [np.array(arr[(arr != pad_value).all(axis=-1)],
                             dtype=dtype)]

    else:
        raise ValueError('Array must be 2- or 3-dimensional')

    a0 = rarr[0].shape[0]
    if all(a.shape[0] == a0 for a in rarr):
        rarr = np.array(rarr, dtype=dtype)

    return rarr


def transform_multioutput(X, y):
    """Converts multi-output data streams for fitting.

    Parameters
    ----------
    X : array-like with shape (n_samples, n_timesteps, n_features)
        Training streams.
    y : array-like with shape (n_samples, n_timesteps)
        Stream target values.

    Returns
    -------
    array-like with shape (n_samples, n_timesteps, n_features)
        Training samples.
    array-like with shape (n_samples,)
        Sample target values.
    """
    X_split, y_split = transform_multioutput_ragged(X, y)
    X_split = to_padded_ndarray(X_split)
    y_split = np.array(y_split, dtype=np.int32)
    return X_split, y_split


def transform_multioutput_ragged(X, y):
    """Converts multi-output data streams for fitting.

    Parameters
    ----------
    X : array-like with shape (n_samples, n_timesteps, n_features)
        Training streams.
    y : array-like with shape (n_samples, n_timesteps)
        Stream target values.

    Returns
    -------
    list with shape (n_samples, n_timesteps, n_features)
        Training samples as an unbalanced list.
    list with shape (n_samples,)
        Sample target values as a list.
    """
    X_split, y_split = ([], [])
    for s, t in zip(X, y):
        index = np.where(np.diff(t) != 0)[0]

        X_split += np.split(s, index)
        y_split += t[index].tolist() + [t[-1]]

    return X_split, y_split


def process_psi(psi, n1, n2):
    """Helper function for processing psi arguments.

    Parameters
    ----------
    psi : float, int, or tuple
        Psi relaxation parameter(s), given as a ratio of time series length.
    n1 : int
        Time series 1 length.
    n2 : int
        Time series 2 length.

    Returns
    -------
    tuple with shape (psi1, psi2)
        Normalized psi relaxation parameters for the given time series lengths.
    """
    # unpack
    if isinstance(psi, tuple):
        psi1, psi2 = psi
    else:
        psi1, psi2 = (psi, psi)

    psi_l = []
    for p, n in [(psi1, n1), (psi2, n2)]:
        if isinstance(p, float):
            if not (0.0 <= p <= 1.0):
                raise ValueError('rel. psi value must be within [0, 1]')
            p = int(round(p * n))
        if p > n:
            raise ValueError('abs. psi value is larger than series length')

        psi_l.append(p)

    return tuple(psi_l)


def relative_score(d, mu, s):
    """Error function for obtaining relative distances. used for ambiguity
    resolution.

    Parameters
    ----------
    d : float
        DTW distance.
    mu : float
        Distribution mean.
    s : float
        Distribution standard deviation.

    Returns
    -------
    float within (0.0, 1.0)
        Relative DTW score based on mean and standard deviation.
    """
    return (0.5 - 0.5 * erf((d - mu) / (np.sqrt(2) * s)))
