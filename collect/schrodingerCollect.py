import scipy.io
from scipy.interpolate import griddata
import numpy as np
import torch

data = scipy.io.loadmat(r'C:\Users\hrida\Physics-Informed-NN\data\NLS.mat')
x = data['x'].flatten()[:, None]
t = data['tt'].flatten()[:, None]
u = data['uu']  
real = np.real(u)
complex = np.imag(u)
modulus = np.sqrt(real**2 + complex**2)

X, T = np.meshgrid(x,t)
train = torch.concat([torch.Tensor(X.flatten()[:, None]), torch.Tensor(T.flatten()[:, None])], 1)
h_star = torch.Tensor(modulus.T.flatten()[:, None])
u_star = torch.Tensor(real.T.flatten()[:, None])
v_star = torch.Tensor(complex.T.flatten()[:, None])

def getData():
    return train, h_star, u_star, v_star