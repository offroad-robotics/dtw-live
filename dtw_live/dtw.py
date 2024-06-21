#!/usr/bin/env python3

import multiprocessing
from functools import partial
from itertools import combinations

import numpy as np

from dtw_live.dtw_c import cost_matrix_c
from dtw_live.utils import process_psi, to_ragged_array


def cost_matrix(s1, s2, psi=0.0, groups=None, use_c=True):
    """Calculate the DTW cost matrix between two templates.

    If `use_c` is `True`, inputs must be a C-contiguous array of type
    `double` (`np.float64`).

    Parameters
    ----------
    s1 : array-like of size (n1, m)
        Time series 1.
    s2 : array-like of size (n2, m)
        Time series 2.
    psi : float or tuple, default=0.0
        Relaxation parameter for ignoring a number of start/end points.
        May be a float between [0,1] or a tuple with values between
        [0,n1] and [0,n2], respectively.
    groups : list or None, default=None
        Specifies index groupings of s1 and s2 which should be grouped
        together using Independent DTW. If this is `None`, Dependent
        DTW is performed
    use_c : bool
        Use the C functions library.

    Returns
    -------
    array-like with shape (n1, n2)
        DTW cost matrix.
    """
    n1 = s1.shape[0]
    n2 = s2.shape[0]

    m = s1.shape[1]
    assert m == s2.shape[1], 'array dim mismatch'

    psi1, psi2 = process_psi(psi, s1.shape[0], s2.shape[0])

    if groups is None:
        mat = _cost_matrix(s1, s2, n1, n2, m, psi1, psi2, use_c)
    else:
        mat = np.zeros((n1, n2), dtype=np.float64)
        for g in groups:
            s1_g = s1[:, g].copy()
            s2_g = s2[:, g].copy()
            m_g = len(g)
            mat += _cost_matrix(s1_g, s2_g, n1, n2, m_g, psi1, psi2, use_c)

    return mat


def _cost_matrix(s1, s2, n1, n2, m, psi1, psi2, use_c):
    """Calculate the DTW cost matrix between two templates.

    Parameters
    ----------
    s1 : array-like of size (n1, m)
        Time series 1.
    s2 : array-like of size (n2, m)
        Time series 2.
    psi1 : int
        Relaxation parameter for time series 1
    psi2 : int
        Relaxation parameter for time series 1

    Returns
    -------
    array-like with shape (n1, n2)
        DTW cost matrix.
    """
    if use_c:
        mat = np.empty((n1, n2), dtype=np.float64)
        cost_matrix_c(s1, s2, n1, n2, m, psi1, psi2, mat)
    else:
        mat = np.empty((n1+1, n2+1), dtype=np.float64)
        # cost boundaries
        mat[psi1:, 0] = np.inf
        mat[0, psi2:] = np.inf
        mat[:psi1+1, 0] = 0.0
        mat[0, :psi2+1] = 0.0

        for i in range(n1):
            for j in range(n2):
                cost = 0.0
                for k in range(m):
                    cost += (s1[i, k] - s2[j, k])**2

                mat[i+1, j+1] = cost + min(mat[i, j],
                                           mat[i, j+1],
                                           mat[i+1, j])

        mat = np.sqrt(mat[1:, 1:])
    return mat


def similarity_cost_matrix(dataset1, dataset2=None, psi=0.0, groups=None,
                           use_c=True, use_mp=True):
    """Calculate the distances between all time series in an iterable. This
    method calculates the cross-similarity cost matrices for all mappings from
    `dataset1` to `dataset2`. If `dataset2` is not given, the cost matrices for
    all combinations of `dataset1` is calculated instead.

    Parameters
    ----------
    dataset : array-like with shape (n_samples1, n_timesteps1, n_features)
        Time series dataset to compare.
    dataset2 : array-like with shape (n_samples2, n_timesteps2, n_features),
    default=None
        Time series dataset to compare. If not `None`, the cross-similarity
        between datasets is calculated.
    psi : float between 0.0-0.3, default=0.0
        Psi relaxation parameter for matching time series start/end points.
        Represented as a percentage of series length.
    use_c : bool, default=True
        Use the C functions library.
    use_mp: bool, defualt=True
        Use python's multiprocessing module to speed up calculation. This
        argument is ignored for sample sizes below 30 (chosen to minize time
        due to multiprocessing overheads).

    Returns
    -------
    list of array-likes
        DTW cost matrices for each combination.
    list of tuples
        Combinations index.
    """
    # ensure input arrays are C-compliant, remove padding
    dataset1 = to_ragged_array(dataset1, dtype=np.float64)

    if dataset2 is None:
        index = list(combinations(np.arange(len(dataset1)), 2))
        dataset2 = dataset1
    else:
        dataset2 = to_ragged_array(dataset2, dtype=np.float64)
        index = [(i, j) for i in range(len(dataset1))
                 for j in range(len(dataset2))]

    if len(index) > 90 and use_mp:
        inputs = [(dataset1[i], dataset2[j]) for i, j in index]
        with multiprocessing.Pool() as pool:
            mats = pool.starmap(partial(cost_matrix,
                                        psi=psi,
                                        groups=groups,
                                        use_c=use_c), inputs)
    else:
        mats = []
        for i, j in index:
            mats.append(cost_matrix(dataset1[i], dataset2[j],
                                    psi=psi,
                                    use_c=use_c))

    return mats, index


def stream_cost_matrix(stream, query, groups=None, use_c=True):
    """Calculates the cost matrix for a given query sequence on a stream. Uses
    :meth:`cost_matrix` with zero-padding (psi_stream == stream_len) to
    replicate the behaviour of :meth:`update_dists_c`.

    This method is a wrapper for `cost_matrix(s1, s2, psi=(len(s1), 0))`.
    Therefore, if `use_c` is `True`, inputs must be a C-contiguous array of
    type `double` (`np.float64`).

    Parameters
    ----------
    stream : array-like with shape (n1, m)
        Stream data.
    tamplate : array-like with shape (n2, m)
        Template data.
    use_c : bool
        Use the C functions library.

    Returns
    -------
    array-like with shape (n1, n2)
        DTW cost matrix.
    """
    return cost_matrix(stream,
                       query,
                       psi=(1.0, 0.0),
                       groups=groups,
                       use_c=use_c)


def stream_similarity_cost_matrix(streams, queries, psi=None, groups=None,
                                  use_c=True, use_mp=True):
    """Calculates the distance matrices for given query sequences on a stream.
    Uses :meth:`cost_matrix` with zero-padding (psi_stream == stream_len) to
    replicate the behaviour of :meth:`update_costs_c`.

    Parameters
    ----------
    streams : array-like with shape (n_samples1, n_timesteps1, n_features)
        Time series streams to compare.
    tamplates : array-like with shape (n_samples2, n_timesteps2, n_features)
        Time series queries to compare.
    psi
        Included for compatibility.
    use_c : bool
        Use the C functions library.
    use_mp : bool
        Use python's multiprocessing module to speed up calculation. This
        argument is ignored for sample sizes below 30 (chosen to minize time
        due to multiprocessing overheads).

    Returns
    -------
    list of array-likes
        DTW cost matrices for each combination.
    list of tuples
        Combinations index.
    """
    return similarity_cost_matrix(streams,
                                  dataset2=queries,
                                  psi=(1.0, 0.0),
                                  groups=groups,
                                  use_c=use_c,
                                  use_mp=use_mp)


def warp_path(mat, psi=0.0):
    """Calculate the warping path for a given cost matrix. Roots of the warping
    path are at the minimum costs of the defined boundaries (based on psi).

    TODO: C implementation

    Parameters
    ----------
    mat : array-like with shape (n1, n2)
        DTW cost matrix.
    psi : float, tuple
        See :meth:`cost_matrix`.

    Returns
    -------
    list of tuples
        Resulting warping path.
    """
    n1, n2 = (mat.shape[0], mat.shape[1])
    psi1, psi2 = process_psi(psi, n1, n2)

    i, j = (n1 - 1, n2 - 1)
    # find starting point based on psi-relaxation
    if psi1 > 0 or psi2 > 0:
        i = i - psi1 + np.argmin(mat[-(psi1+1):, -1])
        j = j - psi2 + np.argmin(mat[-1, -(psi2+1):])
        if mat[i, -1] < mat[-1, j]:
            j = n2 - 1
        else:
            i = n1 - 1

    path = [(i, j)]
    while (i > 0 and j > 0):
        w = np.argmin([mat[i-1, j], mat[i-1, j-1], mat[i, j-1]])
        if w == 0:
            i -= 1
        elif w == 1:
            i -= 1
            j -= 1
        elif w == 2:
            j -= 1
        path.append((i, j))

        if (path[-1][0] == n1) or (path[-1][1] == n2):
            path.pop(0)

    path.reverse()

    return path
