#!/usr/bin/env python3

import numpy as np

from dtw_live.dtw import cost_matrix


def test_cost_matrix_agreement(s1, s2):
    m = cost_matrix(s1, s2, groups=None, use_c=True)
    m_noc = cost_matrix(s1, s2, groups=None, use_c=False)
    assert np.array_equal(m, m_noc), 'DTW cost matrix python/C mismatch'


def test_cost_matrix_groups(s1, s2, groups):
    m = cost_matrix(s1, s2, groups=groups, use_c=True)
    m_g = np.zeros((s1.shape[0], s2.shape[0]))
    for g in groups:
        s1_g = s1[:, g].copy()
        s2_g = s2[:, g].copy()
        m_g += cost_matrix(s1_g, s2_g)
    assert np.array_equal(m, m_g), 'DTW groups/matrix sum mismatch'
