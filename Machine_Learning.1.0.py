#%%
import numpy as np
import keras as keras
from keras.models import Sequential
from keras.layers import Dense,  Dropout
import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
#%%

df = pd.read_csv(r'C:/Users/tjges/OneDrive/Documents/EPO4/Data_25_features/out_window_5.csv')
df = df.drop("SDANN(ECG)", axis=1) #Feature 5 was faulty

#%%
def run_nn(X_train, y_train, X_test, y_test):
    #Creating a simple neural network, sigmoid activation
    input_nodes = X_train.shape[1]
    hidden_layer_1_nodes = 10
    hidden_layer_2_nodes = 20
    hidden_layer_3_nodes = 10
    output_layer = 1

    # initializing a sequential model
    full_model = Sequential()

    # adding layers
    full_model.add(Dense(hidden_layer_1_nodes,input_dim=input_nodes , activation='relu'))
    full_model.add(Dropout(0.1))
    full_model.add(Dense(hidden_layer_2_nodes, activation='relu'))
    full_model.add(Dropout(0.1))
    full_model.add(Dense(hidden_layer_3_nodes, activation='relu'))
    full_model.add(Dropout(0.1))
    full_model.add(Dense(output_layer, activation='sigmoid'))
    full_model.summary()

    # Compiling the ANN
    full_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = full_model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=200, batch_size=10, verbose=2)   
    fig, ax = plt.subplots(1)

    ax.plot(history.history['loss'], label = "loss")
    ax.plot(history.history['val_loss'], label = "validation loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch [n]")
    ax.legend()

    cm = metrics.confusion_matrix(y_test, np.rint(full_model.predict(X_test)))
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'NN Accuracy Score: {0}'.format(history.history["val_accuracy"][-1])
    plt.title(all_sample_title, size = 10);
    return np.rint(full_model.predict(X_test)), history.history["val_accuracy"][-1]
#%%
from functions import smart_train_test_split, load_data, run_svm, normalize_df

train, test = smart_train_test_split(df, 2)

train_norm, test_norm = normalize_df(train, test)


X_train, X_test, Y_train, Y_test = load_data(train_norm, test_norm)

predictions_test, val_acc = run_nn(X_train, Y_train, X_test, Y_test)


# %%
