
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,  Dropout
import keras as keras
import matplotlib.pyplot as plt
import pandas as pd
from hrv import HRV
import pickle
from scipy.signal import butter, iirnotch, lfilter
from sklearn import svm
from sklearn import metrics
import seaborn as sns

def run_nn(X_train, y_train, X_test, y_test, X_self = pd.DataFrame([])):

    self_prediction = 0
    
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
    full_model.add(Dropout(0.2))
    full_model.add(Dense(hidden_layer_2_nodes, activation='relu'))
    full_model.add(Dropout(0.2))
    full_model.add(Dense(hidden_layer_3_nodes, activation='relu'))
    full_model.add(Dropout(0.2))
    full_model.add(Dense(output_layer, activation='sigmoid'))
    full_model.summary()

    # Compiling the ANN
    full_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = full_model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=200, batch_size=25, verbose=2)

    if X_self.empty:
        print("No X_self provided")
    else:
        self_prediction = full_model.predict(X_self)   
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
    return np.rint(full_model.predict(X_test)), history.history["val_accuracy"][-1], np.rint(self_prediction)

def run_svm(X_train, y_train, X_test, y_test):
    model = svm.SVC()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = model.score(X_test, y_test)
    cm = metrics.confusion_matrix(y_test, predictions)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'SVM Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 10);

    return predictions, score

def normalize_df(train, test, self):

    train_id = train.iloc[:,0]
    train_data = train.iloc[:,1:-1]
    train_y = train.iloc[:,-1]
    train_norm =(train_data-train_data.mean())/train_data.std()
    train_norm.insert(0,"ID",train_id)
    train_norm.insert(train_data.shape[1] + 1, "lables", train_y)

    test_id = test.iloc[:,0]
    test_data = test.iloc[:,1:-1]
    test_y = test.iloc[:,-1]
    test_norm =(test_data-train_data.mean())/train_data.std() #Using train mean and STD to normalize also test data
    test_norm.insert(0,"ID",test_id)
    test_norm.insert(test_data.shape[1] + 1, "lables", test_y)

    self_norm = (pd.DataFrame(data=self.values, columns=train.iloc[:,1:-1].columns) - train_data.mean())/train_data.std()

    return train_norm, test_norm, self_norm

def smart_train_test_split(df, test_ID):
    train = []
    test = []

    #for i in range(df["ID"].nunique()):
    for o in range(df.shape[0]):
        if df["ID"][o] == test_ID:
            test.append(np.array(df.iloc[o]))
        else:
            train.append(np.array(df.iloc[o]))
    
    return pd.DataFrame(train), pd.DataFrame(test)

def load_data(train, test):
    X_train = train.iloc[:,1:-1]
    X_test = test.iloc[:,1:-1]
    Y_train = train.iloc[:,-1] - 1 #To obtain labels 0-1 for proper training
    Y_test = test.iloc[:,-1] - 1

    return X_train, X_test, Y_train, Y_test