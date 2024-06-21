#!/usr/bin/env python3

import random
from collections import Counter
from matplotlib import gridspec

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted

import dtw_live.viz as viz
from dtw_live.base import ModelBase
from dtw_live.dtw import (similarity_cost_matrix, warp_path,
                          stream_similarity_cost_matrix)
from dtw_live.dtw_c import update_cost_c
from dtw_live.utils import to_padded_ndarray, transform_multioutput_ragged


class OneVsRestStreamClassifier(BaseEstimator, ClassifierMixin,
                                MultiOutputMixin, ModelBase):
    """Custom Dynamic Time Warping Classifier for continuous time series
    data. Subsequence recognition is formulated as a series of separate binary
    classification problems, with classification based on the mode of
    classification event targets.

    Fitting is performed using subsequences, setting detection thresholds
    based on inputs. Fitted subsequences/thresholds are used as queries for
    input time series data streams.

    Classifier performance (scoring) is evaluated based on subsequence boundary
    conditions.

    Parameters
    ----------
    grace_time : int, default=20
        The number of time steps after a recognition event has occured to
        search for minima/collect additional events. This should be set based
        on the sample rate of your time series.
    metric : str, default='dtw'
        Distance metric to use.
    metric_params : dict or None, default=None
        Dictionary of metric parameters.
        psi : float or int, default=0
            Psi relaxation parameter for time series, represented as either a
            relative size of series length (float between 0 and 1) or as an
            absolute value (int).
        groups : nested list with shape (n_groups,), default=None
            Groups of feature indices for the multidimensional case used to
            apply independent DTW [1]. If `None`, dependent DTW will be used.
        window_size : float or int, default=1.0
            Window size constraint of the Sakoe-Chiba band for DTW. Can be
            represented as either a relative size of series length (float
            between 0 and 1) or as an absolute value (int).
            NOTE: Not implemented.
    feature_names : list of str or None, default=None
        Feature names associated with input data features (optional). Required
        for online recognition.
    target_names : list of str or None, default=None
        Names associated with input target indices given (optional). Required
        for online recognition.

    References
    ----------
    [1] Shokoohi-Yekta, Mohammad et al. "Generalizing DTW to the
    multi-dimensional case requires an adaptive approach." Data mining and
    knowledge discovery vol. 31,1 (2017): 1-31. doi:10.1007/s10618-016-0455-0
    """

    def __init__(self,
                 grace_time=20,
                 metric='dtw',
                 metric_params=None,
                 feature_names=None,
                 target_names=None):
        self.grace_time = grace_time
        self.metric = metric
        self.metric_params = metric_params
        self.feature_names = feature_names
        self.target_names = target_names

    def fit(self, X, y):
        """Fit the classifier with training samples/streams.

        Parameters
        ----------
        X : array-like with shape (n_samples, n_timesteps, n_features) or
        (n_timesteps, n_features)
            Training samples/streams.
        y : array-like with shape (n_samples,), (n_samples, n_timesteps), or
        (n_timesteps,)
            Sample/stream target values.

        Returns
        -------
        KNeighborsStreamClassifier object
            The fitted classifier.
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X, y = check_X_y(X, y,
                         allow_nd=True,
                         multi_output=True,
                         force_all_finite=False)

        # transform data/target streams to ragged query sequences
        if y.ndim == 2:
            X, y = transform_multioutput_ragged(X, y)
            # ignore '-1' as a 'no class' label
            X = [q for q, t in zip(X, y) if t != -1]
            y = [t for t in y if t != -1]

        self.classes_, self.n_classes = np.unique(y, return_counts=True)
        if any(c < 2 for c in self.n_classes):
            raise ValueError('At least 2 samples required per class')

        if self.metric_params is None:
            self.metric_params = {}

        if self.metric == 'dtw':
            mats, ix = similarity_cost_matrix(X, **self.metric_params)
            psi = self.metric_params.get('psi', 0.0)
            if isinstance(psi, tuple):
                raise ValueError('unbalanced psi values not allowed for fit')
            elif (psi > 0.0):
                # trim sequences based on warping path start/end points
                paths = [warp_path(m, psi=psi) for m in mats]
                dists = [m[p[-1]] for m, p in zip(mats, paths)]

                tr = [(0, np.inf) for _ in paths]
                for p, (i, j) in zip(paths, ix):
                    tr[i] = (max(tr[i][0], p[0][0]), min(tr[i][1], p[-1][0]))
                    tr[j] = (max(tr[j][0], p[0][1]), min(tr[j][1], p[-1][1]))

                # apply trimming iff trim is smaller than (1-psi)% of original
                X = [x[ts:te+1, :] if (te - ts) > (1 - psi) * x.shape[0] else x
                     for x, (ts, te) in zip(X, tr)]
                X = to_padded_ndarray(X)
            else:
                dists = [m[-1][-1] for m in mats]
        else:
            raise NotImplementedError('%s' % self.metric)

        # sort distances index by target
        target_dists = [{k: [] for k in self.classes_} for _ in y]
        for d, (i, j) in zip(dists, ix):
            target_dists[i][y[j]].append(d)
            target_dists[j][y[i]].append(d)

        # calculate detection thresholds using min-max difference
        # NOTE: this metric gives bad thresholds when there is overlap
        target_thresholds = []
        for i, d in enumerate(target_dists):
            di_max = max(d[y[i]])
            dj_min = min(min(v) for k, v in d.items() if k != y[i])
            if (di_max > dj_min):
                thr = di_max
            else:
                thr = 0.5 * (dj_min - di_max) + di_max

            target_thresholds.append(thr)

        self.debug_dists = None
        if self.target_names is not None:
            self.debug_dists = [{self.target_names[k]: v for k, v in d.items()}
                                for d in target_dists]

        # model fit params
        self.X_fit = X
        self.y_fit = y
        self.thresholds_fit = np.array(target_thresholds)

        # initialize predict params
        self.reset()
        return self

    def predict(self, X):
        """Predict the class labels for the provided stream data. 

        NOTE: This method resets the current prediction state used by
        :meth:`predict_frame`.

        Parameters
        ----------
        X : array-like with shape (n_timesteps, n_features) or (n_samples,
        n_timesteps, n_features)
            Test streams.

        Returns
        -------
        array-like with shape (n_timesteps,) or (n_samples, n_timesteps)
            Array of predicted class labels and start/end indices.
        """
        check_is_fitted(self)
        X = check_array(X, allow_nd=True, force_all_finite=False)

        if X.ndim == 2:
            X = X[np.newaxis, :]

        y_pred = []
        self._temp_mats = []
        
        for data in X:
            self.reset()

            mats, _ = stream_similarity_cost_matrix(data, self.X_fit,
                                                    **self.metric_params)
            costs = [m[:, -1] for m in mats]
            cost_frames = np.transpose(costs)
            for c in cost_frames:
                self._predict_frame_costs(c)

            y_pred += self.f_events
            
            self._temp_mats.append(mats)

        return y_pred

    def score(self, X, y, sample_weight=None):
        """Get prediction scores for input testing streams. Scoring is based
        on within-bounds events, using F1 as the scoring metric.

        Notes
        -----
        We first check if a predicted event is found within our ground truth
        bounds. If it is, we count it as a true/false positive and move on.
        If not, we count it as a false negative.

        Then, for each ground truth event, we count the number of times a
        prediction has occured within its bounds. We refer to these as
        true/false positive repeated respectively, although they are counted
        as false positive events for probability-like measures.

        Parameters
        ----------
        X : array-like with shape (n_timesteps, n_features) or (n_samples,
        n_timesteps, n_features)
            Test streams.
        y : array-like with shape (n_samples,), (n_samples, n_timesteps), or
        (n_timesteps,)
            Sample/stream target values.
        sample_weight : array-like
            Included for compatibility. This is not used.
        """
        check_is_fitted(self)
        if sample_weight is not None:
            raise NotImplementedError('sample_weight is not implemented')

        if X.ndim == 2:  # single sample
            X = X[np.newaxis, :]

        # nt = len(self.target_names)
        # self.conf_mat = np.zeros((nt + 1, nt + 1), dtype=int)

        TP, FP, FN, TPR, FPR = (0, 0, 0, 0, 0)
        for data, targets in zip(X, y):
            # convert targets to event labels
            index = list(np.where(np.diff(targets) != 0)[0])
            if index[0] != 0:
                index.insert(0, 0)

            ground = []
            for i0, i1 in zip(index, index[1:]):
                t = targets[i1]
                if t != -1:
                    ground.append((t, i0, i1))

            # predict events for test data
            events = self.predict(data)

            """special costs plot
            for mats in self._temp_mats:
                for s1, m in zip(self.X_fit, mats):
                    fig = viz.cost_matrix(
                        np.transpose(m),
                        s2=data,
                        l2='Query Time Series',
                        figsize=(9, 6))
                    axs = fig.axes
                    axs[0].clear()
                    axs[0].set_xticks([])
                    axs[0].set_ylabel('SPRING-DTW Cost')

                    cost = m[:, -1]
                    axs[0].plot(cost)

                    axs[1].clear()
                    axs[1].plot(data[:, [1, 2, 8]])
                    axs[1].set_xlim([0, len(cost)])
                    axs[1].set_xlabel('Samples, Query Time Series')
                    axs[1].set_ylabel('Amplitude')

                    for _, ts, te in ground[:-1]:
                        axs[0].axvspan(ts, te, color=viz.qcgray, alpha=0.2)
                        axs[1].axvspan(ts, te, color=viz.qcgray, alpha=0.2)
                    
                    _, ts, te = ground[-1]
                    axs[0].axvspan(ts, te, color='green', alpha=0.3)
                    axs[1].axvspan(ts, te, color='green', alpha=0.3)

                    # axs[1].set_ylim([-1, 1])
                    axs[1].set_yticks([-1, 0, 1])

                    viz.show()
                    break
                break
            """

            # compare predicted events to ground truth
            for l, ts, te in ground:
                flag = False

                for i, (e, t0, t1) in enumerate(events):
                    # event within bounds (P)
                    if ts < t0 <= te or ts < t1 <= te:
                        if l == e:  # correct class (T)
                            if flag:
                                TPR += 1  # repeated
                            else:
                                TP += 1
                        else:  # wrong class (F)
                            if flag:
                                FPR += 1  # repeated
                            else:
                                FP += 1

                        flag = True
                        events.pop(i)
                        # self.conf_mat[l + 1][e + 1] += 1

                if not flag:  # no event found within bounds (FN)
                    FN += 1
                    # self.conf_mat[l + 1][0] += 1

            # add any events not found within bounds (FP)
            FP += len(events)

            # for (e, _, _) in events:  # add remaining events to null class
            #     self.conf_mat[0][e + 1] += 1

        precision = TP / (TP + FP + FPR + TPR)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        # debug print
        print({'TP': TP, 'FP': FP, 'FN': FN, 'TPR': TPR, 'FPR': FPR, 'f1': f1})
        print(f'precision: {100 * precision:.3f}, recall: {100 * recall:.3f}')
        return f1

    def reset(self):
        """(Re)initialize frame prediction state."""
        check_is_fitted(self)
        self.e_buf = {}
        self.f_events = []
        self.f_cnt = 0
        self.f_costs = np.full(len(self.X_fit), np.inf)
        self.f_dists = [np.full(q.shape[0], np.inf) for q in self.X_fit]

    def predict_frame(self, X):
        """Predict the class labels for the provided stream data.

        Notes
        -----
        This method makes predictions on a frame-by-frame basis, storing
        previous states for future predictions (slow for large datasets). Care
        should be taken to ensure that the prediction state is reset between
        samples (by calling :meth:`reset()`).

        Parameters
        ----------
        X : array-like with shape (n_features,) or (n_timesteps, n_features)
            Test frames.

        Returns
        -------
        array-like with shape (n_timesteps,) or (n_samples, n_timesteps)
            Array of predicted class labels and start/end indices.
        """
        if not (X.flags['C_CONTIGUOUS'] or X.dtype == np.float64):
            X = np.ascontiguousarray(X, dtype=np.float64)

        costs = []
        # TODO: remove nan padding from queries
        for i, query in enumerate(self.X_fit):
            n, m = query.shape
            update_cost_c(X, query, n, m, self.f_dists[i])

            # get current cost (sqrt)
            costs.append(np.sqrt(self.f_dists[i][-1]))

        return self._predict_frame_costs(costs)

    def _predict_frame_costs(self, costs):
        """Process precomputed query costs for a given frame and update event
        predictions. Assumes that costs have the same ordering as fitted
        queries.

        Parameters
        ----------
        costs : array-like with shape (sum(n_classes),)
            Fitted query costs

        Returns
        -------
        y_pred : array-like with shape (sum(n_classes),)
            Predicted events for each template
        """
        if len(costs) != np.sum(self.n_classes):
            raise ValueError('Query costs/count mismatch.')

        for i, cost in enumerate(costs):
            # update event buffer
            if i in self.e_buf:
                start, _, cost_prev = self.e_buf[i]
                if cost < cost_prev:
                    self.e_buf[i] = [start, self.f_cnt, cost]

            # buffer crossing event
            if cost <= self.thresholds_fit[i] < self.f_costs[i]:
                self.e_buf[i] = [self.f_cnt, self.f_cnt, cost]

        # check event durations
        e_durs = []
        for start, _, _ in self.e_buf.values():
            flag = ((self.f_cnt - start) >= self.grace_time)
            e_durs.append(flag)

        # process events if durations elapsed
        y_pred = None
        if e_durs and all(e_durs):
            et = [self.y_fit[i] for i in self.e_buf.keys()]
            # i, e = min(self.e_buf.items(), key=lambda x: x[1][2])

            # get event index + average buffer
            # NOTE: multiple choices are currently selected at random
            counts = Counter(et).most_common(len(self.classes_))
            modes = [l for l, c in counts if c == counts[0][1]]
            e = random.choice(modes)
            # e = modes[0]
            eb = [v for k, v in self.e_buf.items() if self.y_fit[k] == e]
            i0, i1, _ = np.mean(eb, axis=0)

            # if len(multimode) > 1:
            #     for i, l in enumerate(multimode):
            #         t = self.y_fit[l]
            #         print(i, self.target_names[t], int(i0), int(i1))
            # print('-')

            # print([(self.target_names[self.y_fit[k]], v) for k, v in self.e_buf.items() if self.y_fit[k] in multimode])
            # import matplotlib.pyplot as plt
            # x = [self.y_fit[k] for k in self.e_buf.keys() if self.y_fit[k] in multimode]
            # y = [v[2] for k, v in self.e_buf.items() if self.y_fit[k] in multimode]
            # plt.scatter(x, y)
            # viz.show()

            y_pred = (e, int(i0), int(i1))
            self.e_buf.clear()

        self.f_cnt += 1
        self.f_costs = costs

        if y_pred:
            self.f_events.append(y_pred)
        return y_pred

    def _to_dict(self):
        """Return a dict representation of the model (for serialization)
        """
        params = self.get_params()
        fit_params = ['classes_', 'n_classes', 'thresholds_fit',
                      'X_fit', 'y_fit', 'debug_dists']

        for k in fit_params:
            if k in params:
                raise KeyError('key %s already exists in model params' % k)

            params[k] = getattr(self, k)

        for k, v in params.items():
            if isinstance(v, dict):
                params[k] = ModelBase._dict_to_json(v)
            elif isinstance(v, (list, np.ndarray)):
                params[k] = ModelBase._array_to_json(v)

        return params
