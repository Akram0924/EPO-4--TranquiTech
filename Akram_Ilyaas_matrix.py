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
import scipy.signal as signal
import scipy.stats as stats
import scipy.signal
import scipy.ndimage

import numpy as np
from scipy.signal import find_peaks
#%%
def compute_scl(eda_data):
    scl = np.mean(eda_data)
    return scl

def compute_scr_frequency(eda_data, sampling_rate):
    scr_peaks, _ = find_peaks(eda_data, height=0.1)
    scr_frequency = len(scr_peaks) / (len(eda_data) / sampling_rate)
    return scr_frequency

def compute_inter_scr_interval(eda_data, sampling_rate):
    scr_peaks, _ = find_peaks(eda_data, height=0.1)
    inter_scr_intervals = np.diff(scr_peaks) / sampling_rate
    mean_inter_scr_interval = np.mean(inter_scr_intervals)
    return mean_inter_scr_interval

def compute_mean_scr_amplitude(eda_data):
    scr_peaks, _ = find_peaks(eda_data, height=0.1)
    scr_amplitudes = eda_data[scr_peaks]
    mean_scr_amplitude = np.mean(scr_amplitudes)
    return mean_scr_amplitude

def compute_max_scr_amplitude(eda_data):
    scr_peaks, _ = find_peaks(eda_data, height=0.1)
    scr_amplitudes = eda_data[scr_peaks]
    max_scr_amplitude = np.max(scr_amplitudes)
    return max_scr_amplitude

def compute_median_scr_amplitude(eda_data):
    scr_peaks, _ = find_peaks(eda_data, height=0.1)
    scr_amplitudes = eda_data[scr_peaks]
    median_scr_amplitude = np.median(scr_amplitudes)
    return median_scr_amplitude

def compute_num_scrs(eda_data):
    scr_peaks, _ = find_peaks(eda_data, height=0.1)
    num_scrs = len(scr_peaks)
    return num_scrs

def compute_mean_scr_duration(eda_data, sampling_rate):
    scr_peaks, _ = find_peaks(eda_data, height=0.1) 
    scr_durations = np.diff(scr_peaks) / sampling_rate
    mean_scr_duration = np.mean(scr_durations)
    return mean_scr_duration

def compute_auc(eda_data):
    auc = np.sum(np.abs(eda_data))
    return auc
#%%
def ECG_processing(person, time_window):
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
    ECG_subarrays_stress = np.array_split(ECG_stress, len(ECG_stress) // (time_window*fs))
    ECG_samples_stress = np.vstack(ECG_subarrays_stress)

    ECG_baseline = ECG_baseline[:len(ECG_baseline) // (time_window*fs)*(time_window*fs)] 
    ECG_subarrays_baseline = np.array_split(ECG_baseline, len(ECG_baseline) // (time_window*fs))
    ECG_samples_baseline = np.vstack(ECG_subarrays_baseline)

    # #Assigning labels to stress and baseline sets
    ECG_labels_baseline = np.full((ECG_samples_baseline.shape[0], 1), 1) # baseline = 1
    ECG_labels_stress = np.full((ECG_samples_stress.shape[0], 1), 2) # stress = 2

    #Combining stress and baseline sets into one sample set
    ECG_samples = np.vstack((ECG_samples_baseline, ECG_samples_stress))
    #Combinine label set into one
    ECG_labels_tot = np.vstack((ECG_labels_baseline, ECG_labels_stress))

    #Filtering
    nyq = 0.5*fs
    order=5

    # highpass filter
    high=0.5
    high= high/nyq
    b, a = butter(order, high, btype = 'high') # your code here
    ECG_samples_h = lfilter(b,a,ECG_samples)

    # lowpass filter
    low=70
    low= low/nyq
    b, a = butter(order, low, btype = 'low')
    ECG_samples_hl = lfilter(b,a,ECG_samples_h)

    # notch filter
    notch=50
    notch = notch/nyq
    b, a = iirnotch(notch, 30)
    ECG_samples_proc = lfilter(b,a,ECG_samples_hl)

    # detect R-peaks
    from ecgdetectors import Detectors

    rows = ECG_samples_proc.shape[0]
    fs = 700

    def r_peaks_detector(i):
      peak_mat = ECG_samples_proc[i,:]
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

    #Total data matrix including all data
    # data = np.hstack((data_matrix, labels_tot))
    # data = data_matrix

    # #ID array
    # ECG_id_array = np.full((len(ECG_labels_tot), 1), person)
    
    # ECG_data_return = np.hstack((ECG_id_array, data))
    
    
    EMG = chest_data_dict['EMG']

    # Get labels
    labels = obj_data[subject].get_labels()

    # ['transient', 'baseline', 'stress', 'amusement', 'meditation', 'ignore']
    baseline = np.asarray([idx for idx,val in enumerate(labels) if val == 1])

    #Filtering ECG array into the arrays corresdoning to stress and baseline
    EMG_stresslist = [EMG[i] for i in range(len(labels)) if labels[i] == 2]
    EMG_baselinelist = [EMG[i] for i in range(len(labels)) if labels[i] == 1]

    EMG_stress_inter = np.array(EMG_stresslist)
    EMG_baseline_inter = np.array(EMG_baselinelist)

    EMG_stress = EMG_stress_inter.reshape((len(EMG_stresslist)))
    EMG_baseline = EMG_baseline_inter.reshape((len(EMG_baselinelist)))

    #Splitting the stress and baseline array into a fixed time window
    fs = 700

    EMG_stress = EMG_stress[:len(EMG_stress) // (time_window*fs)*(time_window*fs)] 
    EMG_subarrays_stress = np.array_split(EMG_stress, len(EMG_stress) // (time_window*fs))
    EMG_samples_stress = np.vstack(EMG_subarrays_stress)

    EMG_baseline = EMG_baseline[:len(EMG_baseline) // (time_window*fs)*(time_window*fs)] 
    EMG_subarrays_baseline = np.array_split(EMG_baseline, len(EMG_baseline) // (time_window*fs))
    EMG_samples_baseline = np.vstack(EMG_subarrays_baseline)

    # #Assigning labels to stress and baseline sets
    EMG_labels_baseline = np.full((EMG_samples_baseline.shape[0], 1), 1) # baseline = 1
    EMG_labels_stress = np.full((EMG_samples_stress.shape[0], 1), 2) # stress = 2

    #Combining stress and baseline sets into one sample set
    EMG_samples = np.vstack((EMG_samples_baseline, EMG_samples_stress))
    #Combinine label set into one
    EMG_labels_tot = np.vstack((EMG_labels_baseline, EMG_labels_stress))

    #Filtering
    nyq = 0.5*fs
    order=5

    # highpass filter
    high=0.5
    high= high/nyq
    b, a = butter(order, high, btype = 'high') # your code here
    EMG_samples_h = lfilter(b,a,EMG_samples)

    # lowpass filter
    low=70
    low= low/nyq
    b, a = butter(order, low, btype = 'low')
    EMG_samples_hl = lfilter(b,a,EMG_samples_h)

    # notch filter
    notch=50
    notch = notch/nyq
    b, a = iirnotch(notch, 30)
    EMG_samples_proc = lfilter(b,a,EMG_samples_hl)   ####DEZE FEATURE EXTRACTION#####

    rows = EMG_samples_proc.shape[0]
    fs = 700

    EMG_mean_list = [[]]*rows
    for i in range(rows):
      EMG_mean_list[i] = np.mean(EMG_samples_proc[i])
      
    EMG_std_list = [[]]*rows
    for i in range(rows):
      EMG_std_list[i] = np.std(EMG_samples_proc[i])

    EMG_median_list = [[]]*rows
    for i in range(rows):
      EMG_median_list[i] = np.median(EMG_samples_proc[i])
    
    EMG_skew_list = [[]]*rows
    for i in range(rows):
      EMG_skew_list[i] = stats.skew(EMG_samples_proc[i])

    EMG_kurt_list = [[]]*rows
    for i in range(rows):
      EMG_kurt_list[i] = stats.kurtosis(EMG_samples_proc[i])

    EMG_rms_list = [[]]*rows
    for i in range(rows):
      EMG_rms_list[i] = np.sqrt(np.mean(np.square(EMG_samples_proc[i])))

    EMG_zcr_list = [[]]*rows
    for i in range(rows):
      EMG_zcr_list[i] = np.sum(np.diff(np.sign(EMG_samples_proc[i])) != 0) / len(EMG_samples_proc[i])

    EMG_ssc_list = [[]]*rows
    for i in range(rows):
      EMG_ssc_list[i] = np.sum(np.diff(np.sign(np.diff(EMG_samples_proc[i]))) != 0)

    EMG_wavelen_list = [[]]*rows
    for i in range(rows):
      EMG_wavelen_list[i] = waveform_length = np.sum(np.abs(np.diff(EMG_samples_proc[i])))

  
    #ID array
    # EMG_id_array = np.full((len(EMG_labels_tot), 1), person)
    
    # EMG_data_return = np.hstack((EMG_id_array, data)

    EDA = chest_data_dict['EDA']

    # Get labels
    labels = obj_data[subject].get_labels()

    # ['transient', 'baseline', 'stress', 'amusement', 'meditation', 'ignore']
    baseline = np.asarray([idx for idx,val in enumerate(labels) if val == 1])

    #Filtering ECG array into the arrays corresdoning to stress and baseline
    EDA_stresslist = [EDA[i] for i in range(len(labels)) if labels[i] == 2]
    EDA_baselinelist = [EDA[i] for i in range(len(labels)) if labels[i] == 1]

    EDA_stress_inter = np.array(EDA_stresslist)
    EDA_baseline_inter = np.array(EDA_baselinelist)

    EDA_stress = EDA_stress_inter.reshape((len(EDA_stresslist)))
    EDA_baseline = EDA_baseline_inter.reshape((len(EDA_baselinelist)))

    #Splitting the stress and baseline array into a fixed time window
    fs = 700

    EDA_stress = EDA_stress[:len(EDA_stress) // (time_window*fs)*(time_window*fs)] 
    EDA_subarrays_stress = np.array_split(EDA_stress, len(EDA_stress) // (time_window*fs))
    EDA_samples_stress = np.vstack(EDA_subarrays_stress)

    EDA_baseline = EDA_baseline[:len(EDA_baseline) // (time_window*fs)*(time_window*fs)] 
    EDA_subarrays_baseline = np.array_split(EDA_baseline, len(EDA_baseline) // (time_window*fs))
    EDA_samples_baseline = np.vstack(EDA_subarrays_baseline)

    # #Assigning labels to stress and baseline sets
    EDA_labels_baseline = np.full((EDA_samples_baseline.shape[0], 1), 1) # baseline = 1
    EDA_labels_stress = np.full((EDA_samples_stress.shape[0], 1), 2) # stress = 2

    #Combining stress and baseline sets into one sample set
    EDA_samples = np.vstack((EDA_samples_baseline, EDA_samples_stress))
    #Combinine label set into one
    EDA_labels_tot = np.vstack((EDA_labels_baseline, EDA_labels_stress))

    #Filtering
    nyq = 0.5*fs
    order=5

    # highpass filter
    high=0.5
    high= high/nyq
    b, a = butter(order, high, btype = 'high') # your code here
    EDA_samples_h = lfilter(b,a,EDA_samples)

    # lowpass filter
    low=70
    low= low/nyq
    b, a = butter(order, low, btype = 'low')
    EDA_samples_hl = lfilter(b,a,EDA_samples_h)

    # notch filter
    notch=50
    notch = notch/nyq
    b, a = iirnotch(notch, 30)
    EDA_samples_proc = lfilter(b,a,EDA_samples_hl)

    order = 5
    freqs=[0.05]
    sos = scipy.signal.butter(5, freqs, 'high', output='sos', fs=100) #your code here)
    phasic= scipy.signal.sosfiltfilt(sos, EDA_samples_proc)

    rows = EDA_samples_proc.shape[0]
    fs = 700

    def scr_peaks(i):
        Eda_dataset= phasic[i, :]
        scr_peaks, _ = find_peaks(Eda_dataset, height=0.1)
        scr_peaks = np.asarray(scr_peaks)
        return scr_peaks
    
    scr_peaks_list = [[]]*rows       
    for i in range(rows):
     scr_peaks_list[i] = scr_peaks(i)

    scl_list = [[]] * rows
    for i in range(rows):
        # Compute Skin Conductance Level (SCL)
        scl_list[i] = np.mean(compute_scl(phasic[i, :]))

    scr_frequency_list = [[]] * rows
    for i in range(rows):
        # Compute Skin Conductance Response (SCR) Frequency
        scr_frequency_list[i] = np.mean(compute_scr_frequency(phasic[i, :], fs))

    mean_inter_scr_interval_list = [[]] * rows
    for i in range(rows):
        # Compute Mean Inter-SCR Interval
        mean_inter_scr_interval_list[i] = compute_inter_scr_interval(phasic[i, :], fs)

    mean_scr_amplitude_list = [[]] * rows
    for i in range(rows):
        # Compute Mean SCR Amplitude
        mean_scr_amplitude_list[i] = compute_mean_scr_amplitude(phasic[i, :])

    max_scr_amplitude_list = [[]] * rows
    for i in range(rows):
        # Compute Max SCR Amplitude
        max_scr_amplitude_list[i] = np.mean(compute_max_scr_amplitude(phasic[i, :]))

    median_scr_amplitude_list = [[]] * rows
    for i in range(rows):
        # Compute Median SCR Amplitude
        median_scr_amplitude_list[i] = np.mean(compute_median_scr_amplitude(phasic[i, :]))

    num_scrs_list = [[]] * rows
    for i in range(rows):
        # Compute Number of SCRs
        num_scrs_list[i] = compute_num_scrs(phasic[i, :])  # define scr_peaks!!!

    mean_scr_duration_list = [[]] * rows
    for i in range(rows):
        # Compute Mean SCR Duration
        mean_scr_duration_list[i] = compute_mean_scr_duration(phasic[i, :], fs)

    auc_list = [[]] * rows
    for i in range(rows):
        # Compute Total Area under the Curve (AUC)
        auc_list[i] = compute_auc(phasic[i, :])

    #Total data matrix including all data
    # # data = np.hstack((data_matrix, labels_tot))
    # data = data_matrix

    # ID array
    id_array = np.full((len(EDA_labels_tot), 1), person)
  

    listHRV = mean_rr_interval_list
    listHR = HR_list
    listNN20 = NN20_list
    listNN50 = NN50_list
    listSDANN = SDANN_list
    listRMSSD = RMSSD_list
    listSDSD = SDSD_list
    list_EMG_mean = EMG_mean_list
    list_EMG_std = EMG_std_list
    list_EMG_median = EMG_median_list
    list_EMG_skew = EMG_skew_list
    list_EMG_kurt = EMG_kurt_list
    list_EMG_rms = EMG_rms_list
    list_EMG_zcr = EMG_zcr_list
    list_EMG_ssc = EMG_ssc_list
    list_EMG_wavelen = EMG_wavelen_list
    listscl = scl_list
    listscr_frequency = scr_frequency_list
    listmean_inter_scr_interval = mean_inter_scr_interval_list
    listmean_scr_amplitude = mean_scr_amplitude_list
    listmax_scr_amplitude = max_scr_amplitude_list
    listmedian_scr_amplitude = median_scr_amplitude_list
    listnum_scrs = num_scrs_list
    listmean_scr_duration = mean_scr_duration_list
    listauc = auc_list

    matrix_dataa = zip(listHRV, listHR, listNN20, listNN50, listSDANN, listRMSSD, listSDSD,list_EMG_mean, list_EMG_std, list_EMG_median, list_EMG_skew, list_EMG_kurt, list_EMG_rms, list_EMG_zcr, list_EMG_ssc, list_EMG_wavelen,
        listscl, listscr_frequency, listmean_inter_scr_interval, listmean_scr_amplitude,
        listmax_scr_amplitude, listmedian_scr_amplitude, listnum_scrs, listmean_scr_duration, listauc )

    data_matrix = np.array(list(matrix_dataa))

    outputtt = np.hstack((id_array,data_matrix, EMG_labels_tot))

    return outputtt
  
#%%
time_window = 120  # in seconds
start_subject = 2 # first subject
end_subject = 17   # last subject

output = pd.DataFrame()

for i in range(start_subject, end_subject + 1):
    if i == 12:
       continue
    result = pd.DataFrame(ECG_processing(i, time_window))
    output = pd.concat([output, result], ignore_index=True)

output.columns = ['ID', 'RR intervals(ECG)', 'Hearth Rate(ECG)', 'NN20(ECG)', 'NN50(ECG)', 'SDANN(ECG)', 'RMSSD(ECG)', 'SDSD(ECG)', 
              'SCL(EDA)', 'SCR freq.(EDA)', 'SCR mean interval', 'SCR mean amplitude(EDA)', 'SCR max amplitude(EDA)', 'SCR median(EDA)', 'numbers of SCR(EDA)', 'SCR mean duration(EDA)', 'AUC(EDA)',
              'EMG mean', 'EMG std', 'EMG median', 'EMG skewness', 'EMG Kurtosis', 'EMG RMS', 'EMG ZCR', 'EMG SSC', 'Wavelenght(EMG)', 'Label']

output.to_csv('C:/Users/tjges/OneDrive/Documents/EPO4/Data_25_features/out_window_120.csv', index=False)
output
# %%
