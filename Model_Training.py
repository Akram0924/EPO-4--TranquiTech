#%%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras as keras
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#%%

data_x = 10*np.random.rand(6,20)
data_y = np.random.randint(2, size=20)[np.newaxis,:]
data = np.concatenate((data_x,data_y),axis=0)

def load_data(data):
    y = data[-1,:][np.newaxis,:]
    x = data[:-1,:]
    return x, y 

def load_data_T(data):
    y = data[:,-1]
    x = data[:,:-1]
    return x, y 
x, y = load_data(data)
print(x.shape,y.shape)
#%%
#Data must be transposed to fit sklearn train test split
X, y = load_data(data)
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2,shuffle=True, random_state=42)

#standard normalize the data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#%%

#Running PCA to plot two features 
# train pca
pca = PCA(n_components=2) # doing pca and keeping only 5 components
pca = pca.fit(X_train)

# perform pca on features
X_train_pca=pca.transform(X_train);
X_test_pca=pca.transform(X_test)

def plot_2class(x_in,y_in):
    #Create a figure
    plt.figure()
    colors = ['tab:blue', 'tab:orange']
    #iterate over the classes
    for i in range(2):
        #select only the points with class i and plot them in the right colors
        mask = (y_in.astype(int)==i)
        plt.scatter(x_in[0,mask[0,:]], x_in[1,mask[0,:]], marker = 'o', color = colors[i], label = 'y = {}'.format(i))
    #finish the plot
    plt.legend()
    plt.axis('scaled')
    plt.xlabel('PCA Feature 1'); plt.ylabel('PCA Feature 2');
   
plot_2class(X_train_pca.T,y_train.T)
# %%
def run_nn(X_train, y_train, X_test, y_test):
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
    return np.rint(full_model.predict(X_test))
predictions = run_nn(X_train,y_train,X_test,y_test)
#%%
print(predictions)
