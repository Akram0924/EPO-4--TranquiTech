#%%
import pandas as pd
import numpy as np
#%%

def create_array(n_features, n_subjects):
    segments = [27, 24, 30, 145, 210, 204, 207, 198, 196, 157, 140, 123, 145, 167, 180]
    segments = segments[:n_subjects]
    n_segments = sum(segments)
    labels = np.random.randint(2, size=n_segments) + 1
    index = []

    for i in range(n_subjects):
        for o in range(segments[i]):
            index.append(i)

    df = pd.DataFrame(np.random.rand(n_segments, n_features))
    df.insert(n_features,"Label", labels)
    df.insert(0, "ID", index)
    return df

df = create_array(4,3)
#%%
df["ID"][0]
#%%
def train_test_split(df, test_ID):
    train = []
    X_test = []
    Y_train = []
    test = []

    #for i in range(df["ID"].nunique()):
    for o in range(df.shape[0]):
        if df["ID"][o] == test_ID:
            test.append(np.array(df.iloc[o]))
        else:
            train.append(np.array(df.iloc[o]))
    
    return train, test
train, test = train_test_split(df, 1)

# %%
test.length
# %%
