import numpy as np

data_x = 10*np.random.rand(6,30*15)
data_y = np.random.randint(2, size=30*15)[np.newaxis,:]
data = np.concatenate((data_x,data_y),axis=0)