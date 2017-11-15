import numpy as np

from saxskit import saxs_fit, saxs_classify

def test_guinier_porod():
    qvals = np.arange(0.01,1.,0.01)
    Ivals = saxs_fit.guinier_porod(qvals,20,4,120)
    assert isinstance(Ivals,np.ndarray)

if __name__ == '__main__':
    test_guinier_porod()

