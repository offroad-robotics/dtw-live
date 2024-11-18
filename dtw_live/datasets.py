#!/usr/bin/env python3

import json
import os
import random

import numpy as np

from dtw_live.utils import to_padded_ndarray, transform_multioutput


def glob_files(*paths, ftypes=None):
    """Find data at specified paths. May be a directory or path to file.

    Parameters
    ----------
    paths : tuple
        Search paths. May be a directory or file.
    ftypes : list, default=None
        Filetypes to search for. For a list of supported filetypes, see
        :meth:`load_data`.

    Returns
    -------
    list
        paths to globbed files with specified filetypes.
    """
    # find all file paths (check if path is valid/is a directory)
    files = []
    for p in paths:
        if os.path.isfile(p):
            files.append(p)
        elif os.path.isdir(p):
            for f in os.listdir(p):
                if not os.path.isdir(f):
                    files.append(os.path.join(p, f))
        else:
            print('Skipping \'%s\': File or directory not found' % p)
            continue
    if not ftypes:
        return files
    else:
        return [f for f in files if any(f.endswith(t) for t in ftypes)]


def load_data(*paths):
    """Load dataset from specified filepaths. Time series data must contain
    data, labelled subsequences, and share the same header (feature names).

    Currently supported filetypes: `json`.

    Returns
    -------
    array-like with shape (n_samples, n_timesteps, n_features).
        Time series data (padded with NaN values).
    array with shape (n_samples, n_targets, [t, i0, i1])
        Time series targets and associated start/end indices.
    array-like with shape (n_features,)
        Time series feature names.
    array-like with shape (n_targets,)
        Time sereis target names.
    """
    raw_data = []
    for p in paths:
        if p.endswith('.json'):
            with open(p, 'r') as f:
                rd = json.load(f)
            raw_data.append(rd)
        else:
            name = os.path.basename(p)
            print('Skipping \'%s\'. Filetype not supported.' % name)

    if len(raw_data) == 0:
        raise ValueError('No data found.')

    # process raw data and unpack
    processed_data = [_process_data(rd) for rd in raw_data]
    data_p, targets_p, feature_names_p, target_names_p = zip(*processed_data)

    feature_names = np.array(feature_names_p[0])
    if not all(np.array_equal(feature_names, f) for f in feature_names_p[1:]):
        raise ValueError('Feature names mismatch.')

    # convert dataset to padded numpy array (for sklearn compatibility)
    data = to_padded_ndarray(data_p)
    targets = np.full((len(targets_p), data.shape[1]), -1, dtype=np.int32)
    for i, t in enumerate(targets_p):
        targets[i, :t.shape[0]] = t

    # consolidate target names
    target_names = np.unique([i for t in target_names_p for i in t])
    for i, n in enumerate(target_names_p):
        # create ordered subset mapping
        tmap = np.array([np.where(target_names == t)[0][0] for t in n] + [-1])
        targets[i][:] = tmap[targets[i]]

    return data, targets, feature_names, target_names


def _process_data(data):
    """Process raw data dict containing data, targets, target names, and
    feature names.

    Parameters
    ----------
    data : dict
        Dictionary of raw data containing the following keys:
        data : list with shape (n_samples, n_timesteps, n_features)
            Data.
        targets : list with shape (n_samples, n_timesteps,)
            Targets.
        feature_names : list with shape (n_features,)
            Feature names.
        target_names : list with shape (n_targets,)
            Target names.

    Returns
    -------
    array-like with shape (n_timesteps, n_features).
        Data.
    array with shape (n_timesteps,)
        Targets.
    array-like with shape (n_features,)
        Feature names.
    array-like
        Target names.
    """
    # verify file format
    req_keys = ['data', 'targets', 'feature_names', 'target_names']
    missing_keys = [rk for rk in req_keys if rk not in data]
    if missing_keys:
        raise ValueError('Missing keys: %s.' % ', '.join(missing_keys))

    data_ = np.array(data['data'], dtype=np.float64)
    targets = np.array(data['targets'], dtype=np.int32)
    feature_names = np.array(data['feature_names'], dtype=str)
    target_names = np.array(data['target_names'], dtype=str)

    # convert targets to appropriate format
    if targets.ndim > 1 and targets.shape[0] != data_.shape[0]:
        # verify target format
        if targets.ndim > 2 or targets.shape[1] > 3:
            raise ValueError('Invalid targets format')

        if any(i0 > i1 for i0, i1 in targets[:, 1:]):
            raise ValueError('Targets boundaries must be consecutive')

        lex = np.lexsort(np.rot90(targets[:, 1:]))
        ts = targets[lex, 1:]
        if any(i1 > j0 for i1, j0 in zip(ts[:, 1], ts[1:, 0])):
            raise ValueError('Target boundaries must not overlap')

        targets = np.full(data_.shape[0], -1, dtype=np.int32)
        for t, i0, i1 in data['targets']:
            targets[i0:i1+1] = int(t)

    # verify dimensions
    if data_.shape[1] != feature_names.shape[0]:
        raise ValueError('Data/Feature Names mismatch')

    tu = np.unique(targets[targets != -1])
    if tu.shape[0] != target_names.shape[0]:
        raise ValueError('Targets/Target Names mismatch')

    return data_, targets, feature_names, target_names


def filter_data(data, filter_features=None, req_targets=None):
    """Load labelled samples from paths. Filter based on required sensor
    fields and label types.

    Parameters
    ----------
    data : tuple with shape (data, targets, feature_names, target_names)
        Deserialized dataset.
    filter_features : list of strings
        Features to keep.
    req_targets : list of strings
        Required targets, NOTE: non-exclusive for streams.

    Returns
    -------
    array-like with shape (n_samples, n_timesteps, n_features).
        Time series data (padded with NaN values).
    array with shape (n_samples, n_timesteps)
        Time series targets and associated start/end indices.
    array-like with shape (n_features,)
        Time series feature names.
    array-like with shape (n_targets,)
        Time series target names.
    """
    # unpack dataset
    data_, targets, feature_names, target_names = data

    # verify feature filter inputs, get index/filtered list
    if filter_features is None:
        filter_features_index = [i for i, _ in enumerate(feature_names)]
    else:
        invalid_features = [f for f in filter_features if not any(
                            n.startswith(f) for n in feature_names)]
        if invalid_features:
            raise ValueError('Invalid features: ', ', '.join(invalid_features))

        filter_features_index = [i for i, n in enumerate(feature_names)
                                 if any(n.startswith(f) for f in filter_features)]

    feature_names_f = np.array([
        feature_names[i] for i in filter_features_index])

    # verify required target inputs, get index
    if req_targets is None:
        req_targets = target_names
        req_targets_index = np.array([i for i, _ in enumerate(target_names)])
    else:
        invalid_targets = [t for t in req_targets if t not in target_names]
        if invalid_targets:
            raise ValueError('Invalid targets:', ', '.join(invalid_targets))

        req_targets_index = np.array([
            np.where(target_names == rt)[0][0] for rt in req_targets])

    # filter data/targets
    omit_flag = False
    data_f = []
    targets_f_ = []
    for td, tt in zip(data_, targets):
        if not any(t in req_targets_index for t in np.unique(tt)):
            omit_flag = True
            print('Skipped: labels {} not found.'.format(req_targets))
        else:
            data_f.append(td[:, filter_features_index].copy())
            targets_f_.append(tt)

    data_f = to_padded_ndarray(data_f)
    targets_f = np.full((len(targets_f_), data_f.shape[1]), -1, dtype=np.int32)
    for i, t in enumerate(targets_f_):
        targets_f[i, :t.shape[0]] = t

    target_names_f = target_names

    # condense target names/index (if any exclusive to omitted trials)
    if omit_flag:
        tmap = np.array([i if i in np.unique(targets_f) else -1
                         for i, _ in enumerate(target_names)] + [-1])

        target_names_f = np.array([n for i, n in enumerate(target_names)
                                   if i in np.unique(targets_f)])
        for i, t in enumerate(targets_f):
            targets_f[i] = tmap[targets_f[i]]

    return data_f, targets_f, feature_names_f, target_names_f

def load_dataset(*paths, features=None, targets=None):
    """Dataset loading helper. Returns tuple with shape
    (data, targets, feature_names, target_names).
    """

    filepaths = glob_files(*paths)
    data = load_data(*filepaths)
    data_filt = filter_data(data, filter_features=features, req_targets=targets)
    return data_filt

def train_test_split(X, y, test_size=0.2, random_seed=None, shuffle=False):
    """Partition dataset into training and testing streams.

    Parameters
    ----------
    X : array-like with shape (n_samples, n_timesteps, n_features)
        Time series data.
    y : array-like with shape (n_samples, n_timesteps)
        Time series data labels with format (target, start, end).
    test_size : float, default=0.2
        Percentage of the given dataset to use for testing. The rest is used
        for training.
    random_seed : int, default=None
        Random seed for shuffling. If `None`, system time is used as seed.
    shuffle : bool, default=False
    Shuffle samples.

    Returns
    -------
    tuple with shape (X_train, X_test, y_train, y_test)
        Splitted training/testing time series data.
    """
    # Create data partitions that approximate specified test/train sizes
    # return X_train, X_test, y_train, y_test
    raise NotImplementedError('train_test_split not implemented')


def get_samples(data, n_samples=None, random_seed=None, shuffle=False):
    """Collect samples from dataset using label indices.

    Parameters
    ----------
    data : tuples with shape (data, targets, feature_names, target_names)
        Deserialized dataset.
    num_samples : int or None
        Number of samples required per label.
    shuffle : bool
        Shuffle samples. Useful if num_samples is used.

    Returns
    -------
    X : array-like with shape (n_samples, n_timesteps, n_features)
        Sample data.
    y : array-like with shape (n_samples,)
        Sample targets.
    """
    data_, targets, _, _ = data

    # collect data-target pairs
    samples_ = transform_multioutput(data_, targets)
    samples = [(d, t) for d, t in zip(*samples_) if t != -1]

    # shuffle samples
    if shuffle:
        random.seed(random_seed)
        random.shuffle(samples)

    if n_samples is None:
        # return all samples
        X, y = zip(*samples)
    else:
        # collect samples
        X, y = ([], [])
        for d, t in samples:
            count_t = len([i for i in y if i == t])
            if count_t < n_samples:
                X.append(d)
                y.append(t)

        # count samples
        t_unique = set(y)
        for t in t_unique:
            count_t = len([i for i in y if i == t])
            if count_t < n_samples:
                print('Warning: Insufficient samples for \'%s\' (%d/%d).'
                      % (t, count_t, n_samples))

    X = to_padded_ndarray(X)
    y = np.array(y, dtype=np.int32)
    return X, y


def get_feature_groups(feature_names, group_level=None):
    """Generate feature groups index from feature names.

    Parameters
    ----------
    feature_names : array-like with shape (n_features,)
        Time series feature names.
    group_level : int, default=None
        Feature name index level to group by. If none, returns the lowest
        level as a single group.

    Returns
    -------
    list with shape (n_groups,)
        Feature groups index, represented as a ragged 2-d list.

    Example
    -------
    >>> get_feature_groups(['parent1.child1', 'parent1.child2', 'parent2'],
    ...                    group_level=-1)
    array([[0], [1], [2]])
    >>> get_feature_groups(['parent1.child1', 'parent1.child2', 'parent2'],
    ...                    group_level=0)
    array([[0, 1], [2]])
    """

    if group_level is None:
        return [[i for i in range(len(feature_names))]]

    group_names = []
    for fn in feature_names:
        fs = fn.split('.')

        if group_level < 0:
            i = len(fs) + group_level
        else:
            i = group_level

        if not (0 <= i < len(fs)):
            raise ValueError('Invalid group level \'%d\'.' % group_level)

        gn = '.'.join(fs[:i+1])
        if gn not in group_names:
            group_names.append(gn)

    feature_groups = []
    for gn in group_names:
        fg = [i for i, fn in enumerate(feature_names) if fn.startswith(gn)]
        feature_groups.append(fg)

    return feature_groups


def _label_counts(label_data):
    """Get the number of each label from label data as a dict of 
    `label : count` pairs.
    """
    counts = {}
    for l, _, _ in label_data:
        if l in counts:
            counts[l] += 1
        else:
            counts[l] = 1

    return counts
