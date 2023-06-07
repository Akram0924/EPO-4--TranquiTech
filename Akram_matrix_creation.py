#%%
import os
import pickle
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, iirnotch, lfilter
import numpy as np
import matplotlib.pyplot as plt
from ecgdetectors import Detectors
from hrv import HRV
#%%
def processing(person, time_window):
    data_set_path = "C:/Users/tjges/OneDrive/Documents/EPO4/WESAD/WESAD/"
    subject = "S" + str(person)

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
          
            #label = self.data[self.keys[0]]
            assert subject == self.data[self.keys[1]]
            signal = self.data[self.keys[2]]
            wrist_data = signal[self.signal_keys[0]]
            #wrist_ACC = wrist_data[self.wrist_sensor_keys[0]]
            #wrist_ECG = wrist_data[self.wrist_sensor_keys[1]]
            return wrist_data

        def get_chest_data(self):
        
            signal = self.data[self.keys[2]]
            chest_data = signal[self.signal_keys[1]]
            return chest_data

    obj_data = {}
    
    # Accessing class attributes and method through objects
    obj_data[subject] = read_data_of_one_subject(data_set_path, subject)      

    chest_data_dict = obj_data[subject].get_chest_data()
    chest_dict_length = {key: len(value) for key, value in chest_data_dict.items()}

    ECG = chest_data_dict['ECG']

    # Get labels
    labels = obj_data[subject].get_labels()

    # ['transient', 'baseline', 'stress', 'amusement', 'meditation', 'ignore']
    baseline = np.asarray([idx for idx,val in enumerate(labels) if val == 1])

    #Filtering ECG array into the arrays corresdoning to stress and baseline
    ECG_stresslist = [ECG[i] for i in range(len(labels)) if labels[i] == 2]
    ECG_baselinelist = [ECG[i] for i in range(len(labels)) if labels[i] == 1]

    ECG_stress_inter = np.array(ECG_stresslist)
    ECG_baseline_inter = np.array(ECG_baselinelist)

    ECG_stress = ECG_stress_inter.reshape((len(ECG_stresslist)))
    ECG_baseline = ECG_baseline_inter.reshape((len(ECG_baselinelist)))

    #Splitting the stress and baseline array into a fixed time window
    fs = 700

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

    #Filtering
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

    SDANN_list = [[]]*rows
    for i in range(rows):
      SDANN_list[i] = np.mean(hrv.SDANN(r_peaks_list[i]))


    RMSSD_list = [[]]*rows
    for i in range(rows):
      RMSSD_list[i] = np.mean(hrv.RMSSD(r_peaks_list[i]))

    
    SDSD_list = [[]]*rows
    for i in range(rows):
      SDSD_list[i] = np.mean(hrv.SDSD(r_peaks_list[i]))

    #Creating a matrix with each row a sample and each column a feature.
    listHRV = mean_rr_interval_list
    listHR = HR_list
    listNN20 = NN20_list
    listNN50 = NN50_list
    listSDANN = SDANN_list
    listRMSSD = RMSSD_list
    listSDSD = SDSD_list

    matrix_dataa = zip(listHRV, listHR, listNN20, listNN50, listSDANN, listRMSSD, listSDSD)
    data_matrix = np.array(list(matrix_dataa))

    #Total data matrix including all data
    data = np.hstack((data_matrix, labels_tot))

    #ID array
    id_array = np.full((len(labels_tot), 1), person)
    
    data_return = np.hstack((id_array, data))
    
    return data_return

def normalize_df(df):

    df_id = df.iloc[:,0]
    df_data = df.iloc[:,1:-1]
    df_y = df.iloc[:,-1]
    norm_data=(df_data-df_data.mean())/df_data.std()
    norm_data.insert(0,"ID",df_id)
    norm_data.insert(df_data.shape[1] + 1, "lables", df_y)

    return norm_data

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

def load_data(train, test):
    X_train = train.iloc[:,1:-1]
    X_test = test.iloc[:,1:-1]
    Y_train = train.iloc[:,-1]
    Y_test = test.iloc[:,-1]

    return X_train, X_test, Y_train, Y_test

#%%
time_window = 45  # in seconds
start_subject = 2 # first subject
end_subject = 15   # last subject

output = pd.DataFrame()

for i in range(start_subject, end_subject + 1):
    if i == 12:
       continue
    result = pd.DataFrame(processing(i, time_window))
    output = pd.concat([output, result], ignore_index=True)
    #output = output.append(result,ignore_index = True)

output
#%%
#import os  
#os.makedirs('C:/Users/tjges/OneDrive/Documents/EPO4/all_people', exist_ok=True)  
#output.to_csv('C:/Users/tjges/OneDrive/Documents/EPO4/all_people/out.csv') 
# %%
