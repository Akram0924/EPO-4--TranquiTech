#%%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = np.random.rand(6,20)
def load_data(data):
    y = data[-1,:][np.newaxis,:]
    x = data[:-1,:]
    return x, y 

def load_data_T(data):
    y = data[:,-1]
    x = data[:,:-1]
    return x, y 
x, y = load_data_T(data)
print(x.shape,y.shape)
#%%
#Data must be transposed to fit sklearn train test split
X, y = load_data(data)
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2,shuffle=True, random_state=42)

#standard normalize the data

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)
# %%
