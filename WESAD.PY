#%%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt 
from scipy import signal
from scipy.signal import butter, iirnotch, lfilter
from ecgdetectors import Detectors
import pandas as pd

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
ECG = chest_data_dict['ECG']

# %%

# Get labels
labels = obj_data[subject].get_labels() # your code here
baseline = np.asarray([idx for idx,val in enumerate(labels) if val == 1])
plt.plot(labels)

#Filtering ECG array into the arrays corresdoning to stress and baseline
ECG_stresslist = [ECG[i] for i in range(len(labels)) if labels[i] == 2]
ECG_baselinelist = [ECG[i] for i in range(len(labels)) if labels[i] == 1]

ECG_stress_inter = np.array(ECG_stresslist)
ECG_baseline_inter = np.array(ECG_baselinelist)

ECG_stress = ECG_stress_inter.reshape((len(ECG_stresslist)))
ECG_baseline = ECG_baseline_inter.reshape((len(ECG_baselinelist)))

#Splitting the stress and baseline array into a fixed time window
fs = 700
time_window = 60 # in seconds

ECG_stress = ECG_stress[:len(ECG_stress) // (time_window*fs)*(time_window*fs)] 
subarrays_stress = np.array_split(ECG_stress, len(ECG_stress) // (time_window*fs))
samples_stress = np.vstack(subarrays_stress)

ECG_baseline = ECG_baseline[:len(ECG_baseline) // (time_window*fs)*(time_window*fs)] 
subarrays_baseline = np.array_split(ECG_baseline, len(ECG_baseline) // (time_window*fs))
samples_baseline = np.vstack(subarrays_baseline)

#Assigning labels to stress and baseline sets
labels_baseline = np.full((samples_baseline.shape[0], 1), 1) # baseline = 1
labels_stress = np.full((samples_stress.shape[0], 1), 2) # stress = 2

#Combining stress and baseline sets into one sample set
samples = np.vstack((samples_baseline, samples_stress))

#Combinine label set into one
labels_tot = np.vstack((labels_baseline, labels_stress))

#%%
nyq = 0.5*fs
order=5

# highpass filter
high=0.5
high= high/nyq
b, a = butter(order, high, btype = 'high') # your code here
samples_h = lfilter(b,a,samples)

# lowpass filter
low=70
low= low/nyq
b, a = butter(order, low, btype = 'low')
samples_hl = lfilter(b,a,samples_h)

# notch filter
notch=50
notch = notch/nyq
b, a = iirnotch(notch, 30)
samples_proc = lfilter(b,a,samples_hl)
#%%
# detect R-peaks
from ecgdetectors import Detectors

rows = samples_proc.shape[0]
fs = 700

def r_peaks_detector(i):
  peak_mat = samples_proc[i,:]
  detectors = Detectors(fs)
  r_peaks_pan = detectors.pan_tompkins_detector(peak_mat)
  r_peaks_pan = np.asarray(r_peaks_pan)
  return r_peaks_pan

#Empty list to store the peaks of each recording
r_peaks_list = [[]]*rows

for i in range(rows):
  r_peaks_list[i] = r_peaks_detector(i)

#%%
from hrv import HRV
fs = 700
hrv = HRV(fs)

#Calculating the average rr interval in ms
mean_rr_interval_list = [[]]*rows
for i in range(rows):
  mean_rr_interval_list[i] = np.mean(hrv._intervals(r_peaks_list[i]))

#Calculate heart-rates from R peak samples
HR_list = [[]]*rows
for i in range(rows):
    HR_list[i] = np.mean(hrv.HR(r_peaks_list[i]))

#Calculate NN20, the number of pairs of successive NNs that differ by more than 20 ms
NN20_list = [[]]*rows
for i in range(rows):
  NN20_list[i] = hrv.NN20(r_peaks_list[i])

#Calculate NN50, the number of pairs of successive NNs that differ by more than 50 ms
NN50_list = [[]]*rows
for i in range(rows):
  NN50_list[i] = hrv.NN50(r_peaks_list[i])

#Creating a matrix with each row a sample and each column a feature.
listHRV = mean_rr_interval_list
listHR = HR_list
listNN20 = NN20_list
listNN50 = NN50_list

matrix_dataa = zip(listHRV, listHR, listNN20, listNN50)
data_matrix = np.array(list(matrix_dataa)).T

#Total data matrix including all data
data = np.vstack((data_matrix, labels_tot.T))


# %%
print(data)
# %%


