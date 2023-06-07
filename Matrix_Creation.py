#%%
import pandas as pd
import numpy as np
#%%

def create_array(n_features, n_subjects):
    segments = [213, 200, 199, 145, 210, 204, 207, 198, 196, 157, 140, 123, 145, 167, 180]
    segments = segments[:n_subjects]
    n_segments = sum(segments)
    labels = np.random.randint(2, size=n_segments) + 1
    index = []

    for i in range(n_subjects):
        for o in range(segments[i]):
            index.append(i)

    df = pd.DataFrame(np.random.rand(n_segments, n_features)*10)
    df.insert(n_features,"Label", labels)
    df.insert(0, "ID", index)
    return df
df = create_array(4,3)
#%%

def normalize_df(df):

    df_id = df.iloc[:,0]
    df_data = df.iloc[:,1:-1]
    df_y = df.iloc[:,-1]
    norm_data=(df_data-df_data.mean())/df_data.std()
    norm_data.insert(0,"ID",df_id)
    norm_data.insert(df_data.shape[1] + 1, "lables", df_y)

    return norm_data
norm_data = normalize_df(df)
#%%
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
train, test = smart_train_test_split(norm_data, 2)
# %%
def load_data(train, test):
    X_train = train.iloc[:,1:-1]
    X_test = test.iloc[:,1:-1]
    Y_train = train.iloc[:,-1]
    Y_test = test.iloc[:,-1]

    return X_train, X_test, Y_train, Y_test
X_train, X_test, Y_train, Y_test = load_data(train, test)
# %%
from Model_Training import run_nn

predictions_test, predictions_train = run_nn(X_train, Y_train, X_test, Y_test)

# %%
