import os
from matplotlib import pyplot as plt

from ..tools.visualization_tools import doPCA, plot_2d
from ..tools import profiler

def plot_xrsd_fit(sys=None, q=None, I=None, dI=None, show_plot=False):
    mpl_fig = plt.figure() 
    ax_plot = mpl_fig.add_subplot(111)
    I_comp = draw_xrsd_fit(mpl_fig,sys,q,I,dI,show_plot)
    return mpl_fig, I_comp

def draw_xrsd_fit(mpl_fig, sys=None, q=None, I=None, dI=None, show_plot=False):
    ax_plot = mpl_fig.gca()
    ax_plot.clear()
    legend_entries = []
    if q is not None and I is not None:
        ax_plot.semilogy(q,I,lw=2,color='black')
        legend_entries.append('measured')
    I_comp = None
    if sys and q is not None:
        I_comp = sys.compute_intensity(q)
        ax_plot.semilogy(q,I_comp,lw=2,color='red')
        legend_entries.append('computed')
        I_noise = sys.noise_model.compute_intensity(q)
        ax_plot.semilogy(q,I_noise,lw=1) 
        legend_entries.append('noise')
        for popnm,pop in sys.populations.items():
            I_p = pop.compute_intensity(q,sys.sample_metadata['source_wavelength'])
            ax_plot.semilogy(q,I_p,lw=1)
            legend_entries.append(popnm)
    ax_plot.set_xlabel('q (1/Angstrom)')
    ax_plot.set_ylabel('Intensity (counts)')
    ax_plot.legend(legend_entries)
    if show_plot:
        mpl_fig.show()
    return I_comp

def visualize_dataframe(data, labels=['system_class'], features=profiler.profile_keys,
                        use_pca=True, pca_comp_to_use=[0,1], show_plots = False):
    """Makes a labeled scatterplot of data. 

    If use_pca is True,
    PCA will be applied to the data[features], the pca components
    which are specified in pca_comp_to_use will be used for plotting.
    If use_pca is False, the first two features will
    be used for plotting.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels
    labels : list of str
        names of the columns to use for labeling scatterplot points
    features : list of str
        If `use_pca` is True, 
        these are the column names used to evaluate the PCA. 
        If `use_pca` is False, 
        these column names are used directly as axis labels
        (in this case, only the first two entries are used).
    use_pca : bool
        if True, PCA will be applied to features
    pca_comp_to_use : list of int 
        if `use_pca` is True, this is a list of two indices
        indicating which PCs to use as plot axes.
        Each index must be less than the total number of features
    show_plots : bool
        whether or not to show the plots on the display
    """
    if use_pca:
        pca = doPCA(data[features], n_dimensions=len(features))
        transformed_data = pca.transform(data[features])
        for i in range(len(features)):
            data['pc{}'.format(i)] = transformed_data[:,i]
        columns_to_vis = ['pc{}'.format(pca_comp_to_use[0]),'pc{}'.format(pca_comp_to_use[1])]
    else:
        columns_to_vis = features[:2]
    for label in labels:
        plot_2d(data, columns_to_vis, label, False)
    if show_plots:
        plt.show(block=False)
