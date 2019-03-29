import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_2d(data, axis_cols, label, show_plot=True):
    """Make a labeled scatterplot of any two columns of the data.

    Optionally, the plot can be shown on the display 
    or saved at `saving_dir`/`label`_wrt_`axis_cols[0]`_by_`axis_cols[1]`.png.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels
    axis_cols : arr of str
        names of columns to use for the plot axes
    label : str
        column name to use for plot labeling
    show_plot : bool
        whether or not to show the plot on the display
    """
    groups = data.groupby(label)

    fig = plt.figure(figsize=(14,8))
    ax = fig.add_axes([0.1, 0.1, 0.5, 0.75])

    for name, group in groups:
        ax.plot(group[axis_cols[0]],
                group[axis_cols[1]],
                marker='o',alpha=0.5, linestyle='', ms=5, label=name)
        ax.set_xlabel(axis_cols[0])
        ax.set_ylabel(axis_cols[1])
    ax.set_title('By ' + label)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2,
           ncol=1,  borderaxespad=0.)
    if show_plot:
        plt.show(block=False)

def doPCA(data, n_dimensions):
    """Perform principal component analysis on DataFrame `data`.

    Uses Singular Value Decomposition to solve the first
    `n_dimensions` pricipal components of the data.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels
    n_dimensions : int
        number of principal components to keep

    Returns
    -------
    pca : sklearn.decomposition.PCA
        sklearn.decomposition.PCA object
    """
    pca = PCA(n_components=n_dimensions)
    pca.fit(data)
    return pca

