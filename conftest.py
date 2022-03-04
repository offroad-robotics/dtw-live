import pytest
import numpy as np


# sample data
rng = np.arange(0, 3, 0.1, dtype=np.float)
s_sin = np.sin(rng)[:, np.newaxis]
s_cos = np.cos(rng)[:, np.newaxis]


@pytest.fixture
def s1():
    return np.concatenate((s_sin, s_cos), axis=1)


@pytest.fixture
def s2():
    return np.concatenate((s_cos, s_sin), axis=1)


@pytest.fixture
def groups():
    return [[0], [1]]


@pytest.fixture
def data():
    return {
        'data': np.random.rand(30, 4),
        'targets': np.array([[0, 3, 9], [1, 12, 23]]),
        'feature_names': np.array([
            'singlefeature',
            'parent.child1',
            'parent.child2',
            'parent.child3'
        ]),
        'target_names': np.array(['label_1', 'label_2'])
    }


@pytest.fixture
def data_load(data):
    data_ = data['data'][np.newaxis, :]
    targets = np.full(data_.shape[:2], -1, dtype=np.int)
    for t, i0, i1 in data['targets']:
        targets[0, i0:i1+1] = t
    return (
        data_,
        targets,
        data['feature_names'],
        data['target_names']
    )

# psi = (5, 3)
# mat_c = cost_matrix(s1, s2, psi=psi)
# mat_noc = cost_matrix(s1, s2, psi=psi, use_c=False)
# path = warp_path(mat, psi=psi)
# print('Arrays equal:', np.array_equal(p, p_noc))
