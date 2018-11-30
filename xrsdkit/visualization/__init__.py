import os
from matplotlib import pyplot as plt
#from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from ..tools.piftools import get_data_from_Citrination
from ..tools.visualization_tools import doPCA, plot_2d
from ..models import citcl, src_dir, src_dsid_list, testing_data_dir
from ..tools import profiler

default_targets=['system_classification','experiment_id']

def plot_xrsd_fit(sys,q,I,source_wavelength,dI=None,show_plot=False):
    mpl_fig = plt.figure() 
    ax_plot = mpl_fig.add_subplot(111)
    I_comp = draw_xrsd_fit(mpl_fig,sys,q,I,source_wavelength,dI,show_plot)
    return mpl_fig, I_comp

def draw_xrsd_fit(mpl_fig,sys,q,I,source_wavelength,dI=None,show_plot=False):
    ax_plot = mpl_fig.gca()
    ax_plot.clear()
    ax_plot.semilogy(q,I,lw=2,color='black')
    I_comp = sys.compute_intensity(q,source_wavelength)
    ax_plot.semilogy(q,I_comp,lw=2,color='red')
    I_noise = sys.compute_noise_intensity(q)
    ax_plot.semilogy(q,I_noise,lw=1) 
    for popnm,pop in sys.populations.items():
        I_p = pop.compute_intensity(q,source_wavelength)
        ax_plot.semilogy(q,I_p,lw=1)
    ax_plot.set_xlabel('q (1/Angstrom)')
    ax_plot.set_ylabel('Intensity (counts)')
    ax_plot.legend(['measured','computed','noise']+list(sys.populations.keys()))
    if show_plot:
        mpl_fig.show()
    return I_comp

def download_and_visualize(
    source_dataset_ids = src_dsid_list,
    citrination_client = citcl,
    labels = default_targets,
    features = profiler.profile_keys,
    use_pca = True,
    pca_comp_to_use = [0,1],
    show_plots = False,
    save_plots = False,
    saving_dir = testing_data_dir):
    """Download data and plot it with respect two features or principal components.

    This calls xrsdkit.tools.piftools.get_data_from_Citrination()
    to fetch a DataFrame full of xrsd records,
    then xrsdkit.visualization.visualize_dataframe() on that DataFrame.

    Parameters
    ----------
    source_dataset_ids : list of int 
        list of dataset ids to download
    citrination_client : citrination_client.CitrinationClient
        A CitrinationClient instance for downloading data
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
    save_plots : bool
        If True, plots for each label will be saved to
        `saving_dir` / `label` _wrt_(feature_or_PC_0)_by_(feature_or_PC_1).png
    saving_dir : str
        directory to save the plots
    """
    data, _ = get_data_from_Citrination(citrination_client, source_dataset_ids)
    visualize_dataframe(data, labels,features,use_pca, pca_comp_to_use,
                            show_plots, save_plots, saving_dir)

def visualize_dataframe(data,
    labels = default_targets,
    features = profiler.profile_keys,
    use_pca = True,
    pca_comp_to_use = [0,1],
    show_plots = False,
    save_plots = False,
    saving_dir = testing_data_dir):
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
    save_plots : bool
        If True, plots for each label will be saved to
        `saving_dir` / `label` _wrt_(feature_or_PC_0)_by_(feature_or_PC_1).png
    saving_dir : str
        directory to save the plots
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
        plot_2d(data, columns_to_vis, label,
                False, save_plots, saving_dir)
    if show_plots:
        plt.show(block=False)



