#%%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras as keras
import matplotlib.pyplot as plt
#%%

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

#Creating a simple neural network, sigmoid activation
input_nodes = X_train.shape[1]
hidden_layer_1_nodes = 20
hidden_layer_2_nodes = 10
output_layer = 1

# initializing a sequential model
full_model = Sequential()

# adding layers
full_model.add(Dense(hidden_layer_1_nodes,input_dim=input_nodes , activation='relu'))
full_model.add(Dropout(0.1))
full_model.add(Dense(hidden_layer_2_nodes, activation='relu'))
full_model.add(Dropout(0.1))
full_model.add(Dense(output_layer, activation='sigmoid'))
full_model.summary()

# Compiling the ANN
full_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = full_model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=20, batch_size=8, verbose=2)	

fig, ax = plt.subplots(1)

ax.plot(history.history['loss'], label = "loss")
ax.plot(history.history['val_loss'], label = "val_loss")
ax.set_ylabel("Loss")
ax.set_xlabel("Epoch [n]")
ax.legend()


#%%