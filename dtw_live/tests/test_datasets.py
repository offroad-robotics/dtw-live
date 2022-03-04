#!/usr/bin/env python3

import json
import os

import numpy as np
import pytest

from dtw_live.datasets import (glob_files, load_data, filter_data,
                               get_samples, get_feature_groups)
from dtw_live.utils import (to_padded_ndarray)


ROOT = os.path.dirname(os.path.abspath(__file__))
FNAME = 'test_data.json'


@pytest.fixture
def data_fpath(data):
    """Contains setup/takedown."""
    fp_2d = os.path.join(ROOT, FNAME)
    with open(fp_2d, 'w') as f:
        json.dump({k: v.tolist() for k, v in data.items()}, f)

    yield fp_2d
    os.remove(fp_2d)


def test_glob_files_empty():
    paths_empty = glob_files()
    assert paths_empty == []


def test_glob_file_dir(data_fpath):
    paths = glob_files(ROOT, ftypes=['json'])
    assert any(p == data_fpath for p in paths), 'file not found.'


def test_glob_files_file(data_fpath):
    paths = glob_files(data_fpath, ftypes=['json'])
    assert len(paths) <= 1, 'multiple files found for {}'.format(data_fpath)
    assert os.path.basename(paths[0]) == FNAME, 'file not found.'


def test_load_data(data_fpath, data_load):
    data = load_data(data_fpath)
    d, t, fn, tn = data
    print(t.ndim)
    assert d.ndim == 3, 'data should have ndim=3'
    assert t.ndim == 2, 'targets should have ndim=2'
    assert d.shape[0] == 1, 'number of data samples should be 1'
    assert t.shape[0] == 1, 'number of targets samples should be 1'
    assert t.shape[1] == d.shape[1], 'targets should have same size as data'
    assert d.shape[2] == len(fn),\
        'data/feature_names should have the same length'
    assert len(set(tn)) == len(tn), 'targets should be unique'
    assert len(set(t[0, :])) - 1 == len(tn),\
        'set of targets/target_names should have the same length'


def test_filter_data_none(data_load):
    data = filter_data(data_load)
    d, t, _, _ = data
    assert all(np.array_equal(d, dl) for d, dl in zip(data, data_load)),\
        'dataset loaded incorrectly (inconclusive)'


def test_filter_data_features(data_load):
    data = filter_data(data_load, filter_features=['parent'])
    d, _, fn, _ = data
    assert d.shape[2] == len(fn),\
        'data/feature_names should have the same length'


def test_filter_data_targets(data_load):
    data = filter_data(data_load, req_targets=['label_1'])
    assert all(np.array_equal(d, dl) for d, dl in zip(data, data_load)),\
        'dataset loaded incorrectly (inconclusive)'


def test_filter_data_targets_invalid(data_load):
    with pytest.raises(ValueError):
        filter_data(data_load, req_targets=['invalid'])


def test_get_samples(data_load):
    X, y = get_samples(data_load)
    _, t, _, _ = data_load
    assert len(X) == len(y), 'sample data mismatch'
    assert set(y) == set(t[0][t[0] != -1]), 'sample missing dataset labels'


def test_get_feature_groups_default(data_load):
    _, _, fn, _ = data_load
    feature_groups = get_feature_groups(fn, group_level=-1)
    fg = [[0], [1], [2], [3]]
    assert fg == feature_groups, 'feature groups mismatch'


def test_get_feature_groups_zero(data_load):
    _, _, fn, _ = data_load
    feature_groups = get_feature_groups(fn, group_level=0)
    fg = [[0], [1, 2, 3]]
    assert fg == feature_groups, 'feature groups mismatch'
