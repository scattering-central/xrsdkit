import numpy as np

def peaks_by_window(x,y,windowsize=10,thr=0.):
    """Find peaks in x,y data by a window-scanning numerical argument.

    

    Parameters
    ----------
    x : array
        array of x-axis values
    y : array
        array of y-axis values
    windowsize : int
        half-width of window- each point is analyzed
        with the help of this many points in either direction
    thr : float
        for a given point xi,yi, if yi is the maximum within the window,
        the peak is flagged if yi/mean(y_window)-1. > thr

    Returns
    -------
    pk_idx : list of indices where peaks were found
    """
    pk_idx = []
    for idx in range(w,len(y)-w-1):
        pkflag = False
        ywin = y[idx-w:idx+w+1]
        if np.argmax(ywin) == w:
            pkflag = ywin[w]/np.mean(ywin)-1. > thr
        if pkflag:
            pk_idx.append(idx)
    return pk_idx
    

