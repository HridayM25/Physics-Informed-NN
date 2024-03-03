import scipy.io
from scipy.interpolate import griddata
import numpy as np
import torch


data = scipy.io.loadmat('data/burgers_shock.mat')
x = data['x'].flatten()[:, None]
t = data['t'].flatten()[:, None]
usol = np.real(data['usol']).T
X, T = np.meshgrid(x, t)
train = torch.concat([torch.Tensor(X.flatten()[:, None]), torch.Tensor(T.flatten()[:, None])], 1)
X_min = train.min(0)
X_max = train.max(0)

def getData():
    return train, usol, X_min, X_max