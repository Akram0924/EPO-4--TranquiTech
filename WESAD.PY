import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
# %%

class read_data_of_one_subject:
    """Read data from WESAD dataset"""
    def __init__(self, path, subject):
        self.keys = ['label', 'subject', 'signal']
        self.signal_keys = ['wrist', 'chest']
        self.chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        self.wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        #os.chdir(path)
        #os.chdir(subject)
        with open(path + subject +'/'+subject + '.pkl', 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        self.data = data

    def get_labels(self):
        return self.data[self.keys[0]]

    def get_wrist_data(self):
        """"""
        #label = self.data[self.keys[0]]
        assert subject == self.data[self.keys[1]]
        signal = self.data[self.keys[2]]
        wrist_data = signal[self.signal_keys[0]]
        #wrist_ACC = wrist_data[self.wrist_sensor_keys[0]]
        #wrist_ECG = wrist_data[self.wrist_sensor_keys[1]]
        return wrist_data

    def get_chest_data(self):
        """"""
        signal = self.data[self.keys[2]]
        chest_data = signal[self.signal_keys[1]]
        return chest_data
    
data_set_path = "C:/Users/tjges/OneDrive/Documents/EPO4/WESAD/WESAD/" 
subject = 'S3'

# Object instantiation
obj_data = {}
 
# Accessing class attributes and method through objects
obj_data[subject] = read_data_of_one_subject(data_set_path, subject)

chest_data_dict = obj_data[subject].get_chest_data() # your code here
chest_dict_length = {key: len(value) for key, value in chest_data_dict.items()}
print(chest_dict_length)

# %%

# Get labels
labels = obj_data[subject].get_labels() # your code here
baseline = np.asarray([idx for idx,val in enumerate(labels) if val == 1])
plt.plot(labels)
# %%

