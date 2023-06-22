#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from functions import smart_train_test_split, load_data, normalize_df
import seaborn as sns
from sklearn.feature_selection import RFE
#%%
""""
df = pd.read_csv(r'C:/Users/tjges/OneDrive/Documents/EPO4/Data_50_features/out_window_45.csv')
df = df.drop("44", axis=1) #Feature 44 was faulty
df = df.drop("42", axis = 1) #Feature 42 contained a NaN
"""
df = pd.read_csv(r'C:/Users/tjges/OneDrive/Documents/EPO4/Data_25_features/out_window_45.csv')
df = df.drop("SDANN(ECG)", axis=1)

feature_names = df.columns
feature_names = feature_names.drop("ID")
feature_names = feature_names.drop("Label")
#%%
train, test = smart_train_test_split(df, 2)

train_norm, test_norm = normalize_df(train, test)

X_train, X_test, Y_train, Y_test = load_data(train_norm, test_norm)

#feat_importances = pd.Series(model.feature_importances_, index=X.columns)
#feat_importances.nlargest(20).plot(kind='barh')


# define the model
model = RandomForestClassifier()

# fit/train the model on all features
model.fit(X_train, Y_train)

#score
score=model.score(X_test, Y_test)

# get feature importance
importance = model.feature_importances_

#%%
feat_importances = pd.Series(model.feature_importances_, index=feature_names)
feat_importances.nlargest(30).plot(kind='barh')
# %%
df = pd.read_csv(r'C:/Users/tjges/OneDrive/Documents/EPO4/Optimized_50_features/out_window_45.csv')
#df = df.drop("ID", axis = 1)
#df = df.drop("SDANN(ECG)", axis=1)
plt.figure(figsize=(40,28))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

cor_target = abs(cor["Label"])
relevant_features = cor_target[cor_target>0.2]
relevant_features
df[["HRV", "Hearth Rate(ECG)"]].corr()
# %%
