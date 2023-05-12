import numpy as np
data = np.random.rand(6,20)
def load_data(data):
    y = data[-1,:][np.newaxis,:]
    x = data[:-1,:]
    return x, y 

x, y = load_data(data)
print('x =', x,x.shape)
print('y =', y,y.shape)