from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

mnsit_raw = loadmat('mnist-original.mat')
mnsit = {
    'data': mnsit_raw["data"].T,
    'target': mnsit_raw["label"][0]
}

# print(mnsit["data"].shape) # (70000, 784)
# print(mnsit["target"].shape) # (70000,)

x,y = mnsit["data"],mnsit["target"]
# training , testing
# 1-60000 training
# 60001-70000 testing
x_train, x_test, y_train, y_test = x[:60000],x[60000:],y[:60000],y[60000:] # training 60000, testing 10000 

# print(x_train.shape) # (60000, 784) 60000 row, 784 column
# print(x_test.shape) # (10000, 784) 10000 row, 784 column
# print(y_train.shape) # (60000,) 60000 row
# print(y_test.shape) # (10000,) 10000 row