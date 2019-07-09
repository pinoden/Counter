import numpy as np
from scipy.spatial.distance import cdist

def close_pairs(X,max_d):
    d = cdist(X,X)

    I,J = (d<max_d).nonzero()
    IJ  = np.sort(np.vstack((I,J)), axis=0)

    # remove diagonal element
    IJ  = IJ[:,np.diff(IJ,axis=0).ravel()!=0]

    # remove duplicate
    dt = np.dtype([('i',int),('j',int)])
    pairs = np.unique(IJ.T.view(dtype=dt)).view(int).reshape(-1,2)

    return pairs

def test():
    X = np.random.rand(100,2)*20
    p = close_pairs(X,2)
    # print(p)
    from matplotlib import pyplot as plt
    plt.clf()
    plt.plot(X[:,0],X[:,1],'.r')
    plt.plot(X[p,0].T,X[p,1].T,'-b')

# print("test")
# X = np.random.rand(10,2)*20
# print(X,X.shape)
# close_pairs(X,10)
