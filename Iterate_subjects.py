#%%
import pandas as pd
import numpy as np
from functions import smart_train_test_split, load_data, run_svm, normalize_df, run_nn
#%%
def iterate_subjects(start, end):
    df = pd.read_csv(r'C:/Users/tjges/OneDrive/Documents/EPO4/Data_25_features/out_window_120.csv')
    df = df.drop("SDANN(ECG)", axis=1) #Feature 5 was faulty
    NN_score = []
    SVM_score = []

    for i in range(start, end + 1):
        if i == 12:
            continue
        print("subject =", i)

        train, test = smart_train_test_split(df, i)

        train_norm, test_norm = normalize_df(train, test)

        X_train, X_test, Y_train, Y_test = load_data(train_norm, test_norm)

        predictions_test, val_acc = run_nn(X_train, Y_train, X_test, Y_test)
        predictions, svm_acc = run_svm(X_train, Y_train, X_test, Y_test)
        NN_score.append(val_acc)
        SVM_score.append(svm_acc)

    return NN_score, SVM_score
#%%
NN_score, SVM_score = iterate_subjects(2, 17)

# %%
NN_mean = np.mean(NN_score)
NN_std = np.std(NN_score)

SVM_mean = np.mean(SVM_score)
SVM_std = np.std(SVM_score)
# %%
