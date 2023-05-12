import os
import pickle
import numpy as np
import matplotlib.pyplot as plt 
from scipy import signal
from scipy.signal import butter, iirnotch, lfilter
from ecgdetectors import Detectors
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

ecg_base=chest_data_dict['ECG'][baseline,0]
fs=700
# cut a smaller window
ecg=ecg_base[10000:10000+ 10*fs]

t=np.arange(0,ecg.size*(1/fs),(1/fs))
t=t[:ecg.size]

plt.figure(figsize=(12,4))
plt.plot(t,ecg)
plt.xlabel('$Time (s)$') 
plt.ylabel('$ECG$') 

#FILTERING
nyq = 0.5*fs
order=5

# highpass filter
high=0.5
high= high/nyq
b, a = butter(order, high, btype = 'highpass')
ecg_h = lfilter(b,a,ecg)

# lowpass filter
low=70
low= low/nyq
b, a = butter(order, low)
ecg_hl = lfilter(b,a,ecg_h)

# notch filter
notch=50
notch = notch/nyq
b, a = iirnotch(notch, Q = 30, fs = fs)
ecg_hln = lfilter(b,a,ecg_hl)


plt.figure(figsize=(12,4))
plt.plot(t,ecg,label="raw ECG")
plt.plot(t,ecg_hln, label="filtered ECG")
plt.xlabel('$Time (s)$') 
plt.ylabel('$ECG$') 
plt.legend()

# %%
detectors = Detectors(fs)
r_peaks_pan = detectors.pan_tompkins_detector(ecg)
r_peaks_pan = np.asarray(r_peaks_pan)

plt.figure(figsize=(12,4))
plt.plot(ecg)
plt.plot(r_peaks_pan,ecg[r_peaks_pan], 'ro')
#%%
from hrv import HRV   # to import the module
hrv_class = HRV(fs)

feat_nn20=hrv_class.NN20(r_peaks_pan)
feat_nn50=hrv_class.NN50(r_peaks_pan)
feat_rmssd=hrv_class.RMSSD(r_peaks_pan)
# %%




for root, dirs, files in os.walk(data_set_path, topdown=True):
   for name in dirs:
      print(os.path.join(root, name))

      
# %%
