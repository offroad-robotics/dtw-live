#!/usr/bin/env python3

"""
Global Paramters
----------------
target : int
    Target index.
dists : dict
    Target cross-similarity distances.
"""

import numpy as np
from math import copysign, log10


def target_min_max(target, dists):
    """Min-max target function. Weighting is based on the min-max difference
    between the target and nearest class.
    """
    max_l = np.max(dists[target])
    min_d = np.min([np.min(v) for k, v in dists.items() if k != target])

    f = min_d - max_l
    if (min_d < max_l):  # bad
        thr = max_l
    else:
        thr = (max_l + min_d) / 2
    return f, thr


def target_cp_dist(target, dists):
    """Center point distance target function. Weighting is based on the
    attenuated average distance between the target and other classes.
    """
    avg_l = np.mean(dists[target])
    avg_d = [np.mean(v) for k, v in dists.items() if k != target]

    f = np.mean([log10(ad) - log10(avg_l) for ad in avg_d])
    min_d = np.min(avg_d)
    if (min_d < avg_l):  # bad
        thr = avg_l
    else:
        thr = (avg_l + np.min(avg_d)) / 2
    return f, thr


def target_kl_div(target, dists):
    """Kullbach-Liebler Divergence target function.
    """
    raise NotImplementedError()
    avg_l = np.mean(dists[target])
    std_l = np.std(dists[target])

    f = []
    for k, v in dists.items():
        if k == target:
            continue
        avg_d, std_d = np.mean(v), np.std(v)

        A = copysign(1, avg_d - avg_l)
        B = log10((std_d**2 / std_l**2 + std_l**2 / std_d**2 - 2) / 2)
        C = (avg_l - avg_d)**2
        D = (1 / std_l**2 + 1 / std_d**2) / 2

        f.append(A * B + C * D)
    thr = None  # TODO: Calculate intersection point
    return f, thr


def target_erf_int(target, dists):
    """Error Function target function.
    """
    raise NotImplementedError()


def target_min_max_lb(target, dists):
    """Min-max target function with lower bounding. Weighting is based on the
    min-max difference between the target and nearest class. A lower bound is
    set on the given threshold, selecting the minimum distance if another
    class's distance is lower than the target's.
    """
    max_l = np.max(dists[target])
    min_d = np.min([np.min(v) for k, v in dists.items() if k != target])

    f = min_d - max_l
    if (min_d < max_l):  # bad
        thr = max_l
    else:
        thr = min_d  # use minimum inter-class dist
    return f, thr


def target_fixed_dist(target, dists, fixed_dist=2):
    """Fixed distance target function. Weighting is based on the min-max
    difference between the target and nearest class. Thresholds are selected
    as a fixed distance from the target.
    """
    max_l = np.max(dists[target])
    min_d = np.min([np.min(v) for k, v in dists.items() if k != target])

    f = min_d - max_l
    if (min_d < max_l): # bad
        thr = max_l
    else:
        thr = max_l + fixed_dist
    return f, thr
