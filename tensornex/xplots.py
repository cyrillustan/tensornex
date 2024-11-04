import pandas as pd
import xarray as xr
import numpy as np

from matplotlib import gridspec, pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
import seaborn as sns
import scipy.cluster.hierarchy as sch

from .cp import perform_CP, calcR2X
from .tpls import tPLS


def xplot_R2X(data: xr.DataArray, top_rank=12, ax=None, method=perform_CP):
    """Plot increasing rank R2X for CP"""
    assert isinstance(data, xr.DataArray) or isinstance(data, np.ndarray)
    ranks = np.arange(1, min(np.min(data.shape), top_rank) + 1)
    R2Xs = []

    for r in ranks:
        cp = method(data.to_numpy(), r)
        R2Xs.append(calcR2X(cp, data.to_numpy()))

    plt_indep = ax is None
    if plt_indep:
        f = plt.figure(figsize=(5, 4))
        gs = gridspec.GridSpec(1, 1, wspace=0.5)
        ax = plt.subplot(gs[0])

    ax = sns.lineplot(x=ranks, y=R2Xs, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of components")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("R2X")
    ax.set_title("Variance Explained by CP")

    return (f, ax) if plt_indep else ax



def plot_one_heatmap(factor: pd.DataFrame, ax=None, mode_name="",
                     max_entries=False, reorder=False,
                     ytick_classes: pd.DataFrame=None, sort_by_yclasses=False):
    """ Make a heatmap from pandas dataframe """
    # THINK: make sure to consider Tucker and tPLS case

    # Leave entries with low factor values out from the plot
    if max_entries is not False:
        inf_norm = factor.abs().max(axis=1)
        factor = factor.loc[inf_norm.nlargest(max_entries).index]

    # Reorder entries via clustering
    if reorder is None:     # when unspecified, continuous mode names automatically do not reorder
        if any([mode_name.lower().__contains__(keystr) for keystr in ["time", "concentration"]]):
            reorder = False
    if reorder is not False:
        factor = reorder_table(factor)
    if (ytick_classes is not None) and sort_by_yclasses:
        assert ytick_classes.shape == (factor.shape[0], 1)
        factor = factor.loc[ytick_classes.sort_values(ytick_classes.columns[0]).index, :]

    # Actually making the heatmap
    sns.heatmap(
        factor, cmap="PiYG", center=0,
        xticklabels=factor.columns,
        yticklabels=factor.index if factor.shape[0] <= 200 else None,
            # No tick labels if there are too many entries
        cbar=True, vmin=-1.0, vmax=1.0, ax=ax,
    )
    ax.set_xlabel("Components")
    ax.set_title(mode_name)

    # Add subject class color code
    if ytick_classes is not None:
        assert ytick_classes.shape == (factor.shape[0], 1)

        palette = sns.color_palette("Set2", len(np.unique(ytick_classes)))
        class_color_mapping = {cls: color for cls, color in zip(np.unique(ytick_classes), palette)}

        ax.set_yticks(np.arange(ytick_classes.shape[0]) + 0.5)  # Ensure the ytick positions are correct
        for i, label in enumerate(ax.get_yticklabels()):
            label.set_bbox(dict(facecolor=class_color_mapping[ytick_classes.loc[label.get_text()].iloc[0]],
                                edgecolor='none', boxstyle='round,pad=0.1'))

        ax.legend(handles=[mpatches.Patch(color=color, label=cls)
                           for cls, color in class_color_mapping.items()],
                  title='Classes', bbox_to_anchor=(1.05, 1),
                  loc='best')



def xplot_CP(data: xr.DataArray, rank: int, sample_mode=None, sample_class: pd.DataFrame=None, **kwargs):
    """ Run CP on an xarray-formatted data and plot the results. """
    cp = perform_CP(data.to_numpy(), rank, **kwargs)
    ddims = len(data.dims)
    axes_names = list(data.dims)

    factors = [
        pd.DataFrame(
            cp[1][rr],
            columns=[f"{i}" for i in np.arange(1, rank + 1)],
            index=data.coords[axes_names[rr]].values
            if len(data.coords[axes_names[rr]].coords) <= 1
            else [" ".join(ss) for ss in data.coords["Analyte"].values],
        )
        for rr in range(ddims)
    ]

    f = plt.figure(figsize=(8 * ddims, 10))
    gs = gridspec.GridSpec(1, ddims, wspace=0.5)
    axes = [plt.subplot(gs[rr]) for rr in range(ddims)]

    if sample_class is not None:
        assert isinstance(sample_mode, str)
        assert sample_mode in axes_names

    for i in range(ddims):
        plot_one_heatmap(factors[i], ax=axes[i], mode_name=axes_names[i],
                         max_entries=100, reorder=(True if axes_names[i]!=sample_mode else None),
                         ytick_classes=(sample_class if axes_names[i]==sample_mode else None),
                         sort_by_yclasses=True)
    return f, axes, cp



def xplot_tPLS(X: xr.DataArray, Y: xr.DataArray, rank: int, sample_class: pd.DataFrame=None):
    """ Run tPLS on X and Y (disjointed) in xarray format and plot the results. """
    # Check data sanity
    Xddims, Yddims = len(X.dims), len(Y.dims)
    X_axes_names, Y_axes_names = list(X.dims), list(Y.dims)
    assert X_axes_names[0] == Y_axes_names[0]

    # Run tPLS
    plsr = tPLS(rank)
    plsr.fit(X.values, Y.values)

    # Make a list of decomposed factors dataframes
    factors = [
        pd.DataFrame(
            plsr.X_factors[rr] / np.max(plsr.X_factors[rr], axis=0, keepdims=True),
            columns=[f"{i}" for i in np.arange(1, rank + 1)],
            index=X.coords[X_axes_names[rr]].values,
        )
        for rr in range(Xddims)
    ]
    factors += [
        pd.DataFrame(
            plsr.Y_factors[rr] / np.max(plsr.Y_factors[rr], axis=0, keepdims=True),
            columns=[f"{i}" for i in np.arange(1, rank + 1)],
            index=Y.coords[Y_axes_names[rr]].values,
        )
        for rr in range(Yddims)
    ]

    ddims = Xddims + Yddims
    axes_names = X_axes_names + Y_axes_names
    f = plt.figure(figsize=(8 * ddims, 10))
    gs = gridspec.GridSpec(1, ddims, wspace=0.5)
    axes = [plt.subplot(gs[rr]) for rr in range(ddims)]

    # Plotting
    for i in range(ddims):
        plot_one_heatmap(factors[i], ax=axes[i], mode_name=axes_names[i],
                         max_entries=100, reorder=True if (i!=0 and i!=Xddims) else False,
                         ytick_classes=(sample_class if (i==0 or i==Xddims) else None),
                         sort_by_yclasses=True)
    return f, axes, plsr


def reorder_table(df):
    """
    Reorder a table's rows using hierarchical clustering.
    Parameters:
        df (pandas.DataFrame): data to be clustered; rows are treated as samples
            to be clustered
    Returns:
        df (pandas.DataFrame): data with rows reordered via heirarchical
            clustering
    """
    if df.shape[0] <= 1:
        return df
    y = sch.linkage(df.to_numpy(), method="centroid")
    index = sch.dendrogram(y, orientation="right", no_plot=True)["leaves"]
    return df.iloc[index, :]