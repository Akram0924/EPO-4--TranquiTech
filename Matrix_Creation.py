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
df

    
# %%
