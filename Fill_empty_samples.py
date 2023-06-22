
#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

def importData(dataFile):
    data = pd.read_excel(dataFile)
    data = data.drop(data.columns[0], axis=1)
    
    max_timestamp = data['TimeStamp'].max() 
    
    for i in range(max_timestamp):
        print(i, data.loc[i, 'TimeStamp'])
        if i != data.loc[i, 'TimeStamp']:
            if(i == 0):
                new_row = data.loc[i + 1].copy()  # Create a copy of the previous row
            else:
                new_row = data.loc[i - 1].copy()  # Create a copy of the previous row
            
            new_row['TimeStamp'] = i
            data = pd.concat([data.iloc[:i], new_row.to_frame().T, data.iloc[i:]], ignore_index=True)  # Insert the new row
            data.reset_index(drop=True, inplace=True)  # Reset the index

    return data


importedData = importData("C:/Users/tjges/Downloads/P2CorrectTime.xlsx")
importedData.to_csv("C:/Users/tjges/Downloads/P2_filled.xlsx")
print("P1 done")
importedData = importData("P2CorrectTime.xlsx")
importedData.to_csv("FilledData2.csv")
print("P2 done")
importedData = importData("P3CorrectTime.xlsx")
importedData.to_csv("FilledData3.csv")
print("P3 done")
importedData = importData("P4CorrectTime.xlsx")
importedData.to_csv("FilledData4.csv")
print("P4 done")
    
# %%
