#%%
import numpy as np
import keras as keras
import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
#%%

df = pd.read_csv(r'C:/Users/tjges/OneDrive/Documents/EPO4/all_people/out_window_10.csv')
df = df.drop("5", axis=1) #Feature 5 was faulty

#%%


#%%
from functions import smart_train_test_split, load_data, run_nn, run_svm, normalize_df

train, test = smart_train_test_split(df, 4)

train_norm, test_norm = normalize_df(train, test)


X_train, X_test, Y_train, Y_test = load_data(train_norm, test_norm)
predictions_test, predictions_train = run_nn(X_train, Y_train, X_test, Y_test)

errors_test = np.sum(Y_test != np.ravel(predictions_test))
errors_train = np.sum(Y_train != np.ravel(predictions_train))

predictions, score = run_svm(X_train, Y_train, X_test, Y_test)
# %%
input_nodes = X_train.shape[1]
hidden_layer_1_nodes = 10
hidden_layer_2_nodes = 20
hidden_layer_3_nodes = 5
output_layer = 1
