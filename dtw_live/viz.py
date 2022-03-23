#!/usr/bin/env python3

"""
DTW visualizations library.
Queen's color styles are applied from 'qstyles.mplstyle'.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.projections import register_projection

from dtw_live.utils import transform_multioutput, to_padded_ndarray

"""
Queen's color styles
"""
qblue = '#11335D'   # Pantone 295
qlblue = '#004B87'   # Pantone 301 (light blue)
qred = '#9D1939'    # Pantone 187
qgold = '#EEBD31'   # Pantone 124
qcgray = '#686366'  # Pantone Cool Gray 11 (goes with red dominant themes)
qwgray = '#8C7D70'  # Pantone Warm Gray 9 (goes with blue dominant themes)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = 'Palatino'
# plt.rcParams['text.usetex'] = True
plt.rcParams['axes.prop_cycle'] = plt.cycler(
    'color', [qlblue, qred, qgold, qwgray])

# colormap colors
cmap = 'viridis'
pred = '#D52F2A'  # warping path color


class DtwAxes(plt.Axes):
    """Custom matplotlib axes class to make DTW-specific plotting easier.

    Usage
    -----
    ```
    gs = gridspec.GridSpec(1, 3)
    ax = fig.add_subplot(gs[0, 0], projection='dtw')
    ax.plot(x, y, vertical=True, c='r', lw=.6)
    ax.boundary([0, 1], [2, 3], labels=['a', 'b'])
    ax.imshow(m)
    ax.path(p)
    ```
    """
    name = 'dtw'

    def plot(self, *args, vertical=False, **kwargs):
        if len(args) == 0 or len(args) > 2:
            raise ValueError('plot may only contain 2 input args')

        self.vertical = vertical
        if vertical:
            self.set_ylim([0, len(args[0])])
            self.invert_xaxis()

            if len(args) == 1:
                super().plot(args[0], np.arange(len(args[0])), **kwargs)
            elif len(args) == 2:
                super().plot(args[1], args[0], **kwargs)

        else:
            self.set_xlim([0, len(args[0])])
            super().plot(*args, **kwargs)

    def imshow(self, mat, **kwargs):
        super().imshow(mat, origin='lower', aspect='auto', cmap=cmap, **kwargs)

    def boundary(self, *args, labels=None, color='red', label_color='black',
                 ls='dashed', lw=0.6):
        for i, a in enumerate(args):
            if not isinstance(a, (list, tuple)):
                a = [a]

            if self.vertical:
                for b in a:
                    super().axhline(b, color=color, ls=ls, lw=lw)

                if labels:
                    super().text(0.15 * (self.get_xlim()[0] + self.get_xlim()[1]),
                                 a[0],
                                 labels[i],
                                 color=label_color,
                                 verticalalignment='center',
                                 rotation=45)
            else:
                for b in a:
                    super().axvline(b, color=color, ls=ls, lw=lw)

                if labels:
                    super().text(a[-1],
                                 self.get_ylim()[1] + 0.6,
                                 labels[i],
                                 color=label_color,
                                 horizontalalignment='center',
                                 rotation=45)

    def path(self, path, c=pred, ls='solid', lw=1, **kwargs):
        x = [i for i, _ in path]
        y = [j for _, j in path]

        super().plot(y, x, c=c, ls=ls, lw=lw, **kwargs)

    def set_title(self, label, vertical=False):
        if vertical:
            super().set_ylabel(label, fontsize=11)
        else:
            super().set_title(label, fontsize=11)


# register custom axes
register_projection(DtwAxes)


def cost_matrix(mat, path=None, s1=None, l1=None, s2=None, l2=None,
                plot_trim=False, **kwargs):
    """Plot the DTW cost matrix for two sequences.
    """
    fig = plt.figure(figsize=kwargs.get('figsize', None))
    # setup gridspec based on parameters
    nrows = 1 if s2 is None else 2
    ncols = 1 if s1 is None else 2

    height_ratios = None if s2 is None else [5, 1]
    width_ratios = None if s1 is None else [1, 6]

    gs = gridspec.GridSpec(nrows, ncols,
                           height_ratios=height_ratios,
                           width_ratios=width_ratios)

    # plot matrix
    ax_mat = fig.add_subplot(gs[0, ncols-1],
                             projection='dtw')
    ax_mat.imshow(mat)

    # plot path
    if path is not None:
        ax_mat.path(path, lw=1.5)

    # plot sequence 1 (left)
    if s1 is not None:
        ax_mat.get_yaxis().set_ticks([])
        ax_s1 = fig.add_subplot(gs[0, 0],
                                projection='dtw')
        ax_s1.plot(s1, vertical=True)
        ax_s1.autoscale(enable=True, axis='y', tight=True)

        # add trim boundaries
        if plot_trim and path is not None:
            ax_s1.boundary((path[0][0], path[-1][0]+1), lw=0.6)

        # add label
        if l1 is not None:
            ax_s1.set_title('Samples, ' + l1, vertical=True)
        else:
            ax_s1.set_title('Samples, Time Series 1', vertical=True)
    
    elif l1 is not None:
        ax_mat.set_ylabel('Samples, ' + l1)
    else:
        ax_mat.set_ylabel('Samples, Time Series 1')
        

    # plot sequence 2 (bottom)
    if s2 is not None:
        ax_mat.get_xaxis().set_ticks([])
        ax_s2 = fig.add_subplot(gs[nrows-1, ncols-1],
                                projection='dtw')
        ax_s2.plot(s2)
        ax_s2.autoscale(enable=True, axis='x', tight=True)

        # add trim boundaries
        if plot_trim and path is not None:
            ax_s2.boundary((path[0][1], path[-1][1]+1), lw=0.6)

        # add label
        if l2 is not None:
            ax_s2.set_xlabel('Samples, ' + l2)
        else:
            ax_s2.set_xlabel('Samples, Time Series 1')
        
    elif l2 is not None:
        ax_mat.set_xlabel('Samples, ' + l2)
    else:
        ax_mat.set_ylabel('Samples, Time Series 2')

    if s1 is not None and s2 is not None:
        ax_ylab = fig.add_subplot(gs[nrows-1, 0])
        ax_ylab.text(0.5, 0.5, 'Amplitude', ha='center', va='center')
        ax_ylab.axis('off')
    elif s1 is not None:
        ax_s1.set_xlabel('Amplitude')
    elif s2 is not None:
        ax_s2.set_ylabel('Amplitude')

    fig.tight_layout()
    return fig


@DeprecationWarning
def stream_distances(*dists, thresholds=None, events=None, stream=None,
                     label=None, boundary=None, title=None,
                     plot_boundary='stream', **kwargs):
    """Plot stream distances as continuous output (similar to SPRING-DTW).
    """
    fig = plt.figure(figsize=kwargs.get('figsize', None))
    if title is not None:
        fig.suptitle(title)

    # setup gridspec based on parameters
    nrows = 2 if stream is not None else 1
    height_ratios = [1, 5] if stream is not None else None
    gs = gridspec.GridSpec(nrows, 1, height_ratios=height_ratios)

    # plot distances
    ax_dists = fig.add_subplot(gs[nrows-1, 0],
                               projection='dtw')
    for d in dists:
        ax_dists.plot(d)

    if thresholds is not None:
        if isinstance(thresholds, (list, np.ndarray)):
            for thr in thresholds:
                ax_dists.axhline(thr, ls='--', lw=0.5, c='k')
        else:
            ax_dists.axhline(thresholds, ls='--', lw=0.5, c='k')
    if events is not None:
        for e in events:
            ax_dists.axvline(e, ls='-.', lw=1, c='g')

    # plot stream sequence (top)
    if stream is not None:
        ax_stream = fig.add_subplot(gs[0, 0],
                                    projection='dtw')
        ax_stream.plot(stream)

        # add labels/boundaries
        if boundary is not None:
            if plot_boundary == 'all':
                ax_stream.boundary(*boundary, labels=label)
                ax_dists.boundary(*boundary, labels=None)
            elif plot_boundary == 'stream':
                ax_stream.boundary(*boundary, labels=label)
            if plot_boundary == 'distances':
                ax_dists.boundary(*boundary, labels=label)

    fig.tight_layout()


def sequence_alignment(path, s1, s2, l1=None, l2=None, feature_index=None,
                       resolution=3, **kwargs):
    """Plot sequence alignment for the specified time series/feature index
    """
    if feature_index is None:
        feature_index = np.arange(s1.shape[1] if (len(s1.shape) > 1) else 1)

    n_inputs = len(feature_index)
    nrows = int(np.ceil(np.sqrt(n_inputs)))
    ncols = int(np.ceil(n_inputs / nrows))

    fig = plt.figure(figsize=kwargs.get('figsize', None))

    for i, ix in enumerate(feature_index):
        s1_ix = s1[:, ix] if (len(s1.shape) > 1) else s1
        s2_ix = s2[:, ix] if (len(s1.shape) > 1) else s2

        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.plot(s1_ix, label=l1)
        ax.plot(s2_ix, label=l2)

        for i, (p1, p2) in enumerate(path):
            if (i % resolution == 0):  # every resolution-th line (for speed)
                ax.plot([p1, p2], [s1_ix[p1], s2_ix[p2]],
                        ls='--', c=qcgray, lw=0.5)

    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude')
    ax.autoscale(enable=True, axis='x', tight=True)

    if l1 is not None and l2 is not None:
        ax.legend([l1, l2], loc="lower center",
                  bbox_to_anchor=(0.5, -0.25), ncol=2)
        plt.subplots_adjust(bottom=0.2)

    fig.tight_layout()
    return fig


def class_distances(dists_dicts, thresholds=None, **kwargs):
    """Display the calculated distances between a given template as box plots,
    grouped by class. Data must be formatted accordingly.

    Parameters
    ----------
    dists_dicts : list of dicts with target-distances pairs
        Distance combinations for individual templates.
    """
    nrows = int(np.ceil(np.sqrt(len(dists_dicts))))
    ncols = int(np.ceil(len(dists_dicts) / nrows))

    fig = plt.figure(figsize=kwargs.get('figsize', None))
    sharedClass = all(dists_dicts[0][0] == a[0] for a in dists_dicts[1:])

    if sharedClass:
        fig.suptitle(dists_dicts[0][0])

    for i, (l, d) in enumerate(dists_dicts):
        ax = fig.add_subplot(nrows, ncols, i+1)
        if not sharedClass:
            ax.set_title(l)

        xlabels = list(d.keys())
        vix = np.where(np.array(xlabels) == l)[0] + 1
        ax.axvline(vix - 0.35, ls='--', lw=0.5, c='black')
        ax.axvline(vix + 0.35, ls='--', lw=0.5, c='black')
        # ax.boxplot(d.values())
        for j, v in enumerate(d.values()):
            ax.scatter([j for _ in v], v, s=10)

        if thresholds is not None:
            ax.axhline(thresholds[i], ls='--', lw=0.6, c='black')

        ax.set_xticklabels(xlabels)
        plt.setp(ax.get_xticklabels(), rotation=45, fontsize=8,
                 ha='right', rotation_mode='anchor')

    fig.tight_layout()


def samples_plot(X, y, target_names=None, legend=None, fig_multi=True,
                 **kwargs):
    """Plot all raw samples and organize into rows/figures by target.
    """
    if isinstance(y, np.ndarray) and y.ndim == 2:
        X, y = transform_multioutput(X, y)
        X = to_padded_ndarray(X[y != -1])
        y = np.array(y[y != -1], dtype=np.int)

    if target_names is not None:
        y = [target_names[i] for i in y]

    unique, counts = np.unique(y, return_counts=True)

    # plot everything in a single figure
    if not fig_multi:
        fig = plt.figure(figsize=kwargs.get('figsize', None))
        nrows = len(unique)
        ncols = max(counts)

    for i, (u, n) in enumerate(zip(unique, counts)):
        X_y = [s for s, l in zip(X, y) if l == u]

        # plot everything in separate figures
        if fig_multi:
            nrows = int(np.ceil(np.sqrt(n)))
            ncols = int(np.ceil(n / nrows))
            fig = plt.figure(figsize=kwargs.get('figsize', None))
            fig.suptitle(u)

        for j, s in enumerate(X_y):
            if not fig_multi:
                ax = fig.add_subplot(nrows, ncols, i*ncols + j + 1)
                if j == 0:
                    ax.set_ylabel(u)
            else:
                ax = fig.add_subplot(nrows, ncols, j + 1)
            ax.plot(s)
            ax.set_ylim((-1, 1))

        # place legend on first plot
        if legend is not None and (fig_multi or i == 0):
            ax.legend(legend, bbox_to_anchor=(1.05, 1), loc='upper left')
        # if fig_multi:
        #     fig.tight_layout()

    # if not fig_multi:
    #     fig.tight_layout()


def show(close_sig=True, timeout=0):
    """Call `plt.show()` with optional `plt.close('all')` handlers. 

    Parameters
    ----------
    close_sig : bool
        If `True`, sets up an event handler to close all windows when the
        `escape` key is pressed.
    timeout : float
        If non-zero, plots will close after a specified time.
    """

    if close_sig:
        def on_key(event):
            if event.key == 'escape':
                plt.close('all')
            elif event.key == 'ctrl+c':
                plt.close('all')
                exit()  # bad practice, but good for killing plots in a loop

        for i in plt.get_fignums():
            plt.figure(i).canvas.mpl_connect('key_press_event', on_key)

    if timeout != 0:
        plt.show(block=False)
        plt.pause(timeout)
        plt.close('all')
    else:
        plt.show()


def savefig(fname):
    """Calls `plt.savefig()` with preferred options.
    """

    plt.savefig(fname)
