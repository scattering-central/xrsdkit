import numpy as np

def peaks_by_window(x,y,w=10,thr=0.):
    """Find peaks in x,y data by a window-scanning.

    TODO: introduce window shapes and make use of the x-values.

    Parameters
    ----------
    x : array
        array of x-axis values
    y : array
        array of y-axis values
    w : int
        half-width of window- each point is analyzed
        with the help of this many points in either direction
    thr : float
        for a given point xi,yi, if yi is the maximum within the window,
        the peak is flagged if yi/mean(y_window)-1. > thr

    Returns
    -------
    pk_idx : list of int
        list of indices where peaks were found
    pk_confidence : list of float
        confidence in peak labeling for each peak found 
    """
    pk_idx = []
    pk_confidence = []
    for idx in range(w,len(y)-w-1):
        pkflag = False
        ywin = y[idx-w:idx+w+1]
        if np.argmax(ywin) == w:
            conf = ywin[w]/np.mean(ywin)-1.
            pkflag = conf > thr
        if pkflag:
            pk_idx.append(idx)
            pk_confidence.append(conf)

    #from matplotlib import pyplot as plt
    #plt.figure(2)
    #plt.plot(x,y)
    #for ipk,cpk in zip(pk_idx,pk_confidence):
    #    qpk = x[ipk]
    #    Ipk = y[ipk]
    #    print('q: {}, I: {}, confidence: {}'.format(qpk,Ipk,cpk))
    #    plt.plot(qpk,Ipk,'ro')
    #plt.show()

    return pk_idx,pk_confidence
    

