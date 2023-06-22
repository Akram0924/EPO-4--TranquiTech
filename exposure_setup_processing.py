#%%
import os
import csv
import pickle
import scipy.ndimage
from hrv import HRV
import numpy as np
from scipy import stats
import pandas as pd
from scipy import signal
from scipy.signal import butter, iirnotch, lfilter
import matplotlib.pyplot as plt
from ecgdetectors import Detectors
import scipy.signal as signal
import scipy.stats as stats
import biosppy.signals.ecg as ecg
from scipy.signal import welch
from ecgdetectors import Detectors
from scipy.signal import find_peaks
#%%
def compute_scr_rise_time(eda_data, sampling_rate):
    threshold = 0.1 * np.max(eda_data)
    peaks, _ = find_peaks(eda_data, height=threshold)
    if len(peaks) < 2:
        return np.nan
    return (peaks[1] - peaks[0]) / sampling_rate

def compute_scr_recovery_time(eda_data, sampling_rate):
    threshold = 0.1 * np.max(eda_data)
    peaks, _ = find_peaks(eda_data, height=threshold)
    if len(peaks) < 2:
        return np.nan
    return (peaks[-1] - peaks[-2]) / sampling_rate

def compute_scr_half_recovery_time(eda_data, sampling_rate):
    threshold = 0.5 * np.max(eda_data)
    peaks, _ = find_peaks(eda_data, height=threshold)
    if len(peaks) < 2:
        return np.nan
    return (peaks[-1] - peaks[-2]) / sampling_rate

def compute_scr_sensitivity(eda_data, tonic_data):
    mean_tonic = np.mean(tonic_data)
    if mean_tonic == 0:
        return np.nan
    return np.mean(eda_data) / mean_tonic

def compute_scr_frequency_band(eda_data, sampling_rate):
    # Compute SCR Frequency Band
    low_freq = 0.08  # Lower frequency threshold for SCR
    high_freq = 0.5  # Upper frequency threshold for SCR
    freqs, psd = welch(eda_data, fs=sampling_rate)
    scr_band = psd[(freqs >= low_freq) & (freqs <= high_freq)]
    return np.sum(scr_band)

def compute_scr_energy(eda_data):
    # Compute SCR Energy
    return np.sum(np.square(eda_data))

def compute_scr_slope(eda_data, sampling_rate):
    # Compute SCR Slope
    time = np.arange(len(eda_data)) / sampling_rate
    slope, _ = np.polyfit(time, eda_data, deg=1)
    return slope

def compute_scr_amplitude_variation(eda_data):
    # Compute SCR Amplitude Variation
    return np.std(eda_data)

def compute_scr_eda_correlation(eda_data, tonic_data):
    # Compute SCR-EDA Correlation
    return np.corrcoef(eda_data, tonic_data)[0, 1]

def compute_scr_area_ratio(eda_data, tonic_data):
    # Compute SCR Area Ratio
    scr_area = np.sum(eda_data)
    tonic_area = np.sum(tonic_data)
    if tonic_area == 0:
        return np.nan
    return scr_area / tonic_area

def compute_scr_frequency_slope(eda_data, sampling_rate):
    # Compute SCR Frequency Slope
    freqs, psd = welch(eda_data, fs=sampling_rate)
    freq_range = freqs[1:]  # Exclude 0 Hz
    psd_diff = np.diff(psd)
    if np.all(psd_diff == 0):
        return np.nan
    slope, _ = np.polyfit(freq_range, psd_diff, deg=1)
    return slope

def compute_scr_time_ratio(eda_data, tonic_data, sampling_rate):
    # Compute SCR Time Ratio
    scr_duration = compute_scr_recovery_time(eda_data, sampling_rate)
    tonic_duration = len(tonic_data) / sampling_rate
    if tonic_duration == 0:
        return np.nan
    return scr_duration / tonic_duration

def compute_scr_area_time_ratio(eda_data, tonic_data, sampling_rate):
    # Compute SCR Area-Time Ratio
    scr_area = np.sum(eda_data)
    scr_duration = compute_scr_recovery_time(eda_data, sampling_rate)
    tonic_area = np.sum(tonic_data)
    tonic_duration = len(tonic_data) / sampling_rate
    if tonic_duration == 0:
        return np.nan
    return (scr_area * tonic_duration) / (tonic_area * scr_duration)

def compute_phasic_tonic_ratio(eda_data, tonic_data):
    # Compute Phasic-Tonic Ratio
    phasic_area = np.sum(eda_data)
    tonic_area = np.sum(tonic_data)
    if tonic_area == 0:
        return np.nan
    return phasic_area / tonic_area

def compute_phasic_activity(eda_data):
    # Compute Phasic Activity
    return np.sum(eda_data)

def compute_tonic_activity(tonic_data):
    # Compute Tonic Activity
    return np.sum(tonic_data)
#%%
def compute_emg_max(EMG_samples_proc):
    emg_max_list = [np.max(sample) for sample in EMG_samples_proc]
    return emg_max_list

def compute_emg_min(EMG_samples_proc):
    emg_min_list = [np.min(sample) for sample in EMG_samples_proc]
    return emg_min_list

def compute_emg_range(EMG_samples_proc):
    emg_range_list = [np.max(sample) - np.min(sample) for sample in EMG_samples_proc]
    return emg_range_list

def compute_emg_power(EMG_samples_proc):
    emg_power_list = [np.sum(np.square(sample)) / len(sample) for sample in EMG_samples_proc]
    return emg_power_list

def compute_emg_crest_factor(EMG_samples_proc):
    emg_crest_factor_list = [np.max(np.abs(sample)) / np.sqrt(np.mean(np.square(sample))) for sample in EMG_samples_proc]
    return emg_crest_factor_list

def compute_emg_variance(EMG_samples_proc):
    emg_variance_list = [np.var(sample) for sample in EMG_samples_proc]
    return emg_variance_list

def compute_emg_peak_to_peak(EMG_samples_proc):
    emg_peak_to_peak_list = [np.max(sample) - np.min(sample) for sample in EMG_samples_proc]
    return emg_peak_to_peak_list

def compute_emg_crest(EMG_samples_proc):
    emg_crest_list = [np.max(np.abs(sample)) for sample in EMG_samples_proc]
    return emg_crest_list

def compute_emg_abs_integral(EMG_samples_proc):
    emg_abs_integral_list = [np.sum(np.abs(sample)) for sample in EMG_samples_proc]
    return emg_abs_integral_list

def compute_emg_squared_integral(EMG_samples_proc):
    emg_squared_integral_list = [np.sum(np.square(sample)) for sample in EMG_samples_proc]
    return emg_squared_integral_list

def compute_emg_mean_absolute_deviation(EMG_samples_proc):
    emg_mean_absolute_deviation_list = [np.mean(np.abs(sample - np.mean(sample))) for sample in EMG_samples_proc]
    return emg_mean_absolute_deviation_list

def compute_emg_zero_crossing_rate(EMG_samples_proc):
    emg_zero_crossing_rate_list = [np.sum(np.diff(np.sign(sample)) != 0) / len(sample) for sample in EMG_samples_proc]
    return emg_zero_crossing_rate_list

def compute_emg_mean_list(EMG_samples_proc):
    emg_mean_list = [np.mean(sample) for sample in EMG_samples_proc]
    return emg_mean_list

def compute_emg_std_list(EMG_samples_proc):
    emg_std_list = [np.std(sample) for sample in EMG_samples_proc]
    return emg_std_list

def compute_emg_skew_list(EMG_samples_proc):
    emg_skew_list = [stats.skew(sample) for sample in EMG_samples_proc]
    return emg_skew_list

def compute_emg_kurt_list(EMG_samples_proc):
    emg_kurt_list = [stats.kurtosis(sample) for sample in EMG_samples_proc]
    return emg_kurt_list
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

def compute_scr_tonic_ratio(pha_data, ton_data):
    # Compute SCR Tonic Ratio
    scr_area = np.sum(pha_data)
    tonic_area = np.sum(ton_data)
    if tonic_area == 0:
        return np.nan
    return scr_area / tonic_area

def compute_scr_area_peaks(pha_data):
    # Compute SCR Area Peaks
    peaks, _ = find_peaks(pha_data)
    return np.sum(pha_data[peaks])
#%%
data = []
data_person = pd.read_csv('C:/Users/tjges/OneDrive/Documents/EPO4/Exposure Data/P2.csv')

#%%
def processing(time_window):

    fs = 1000

    ECG_data_pd = data_person['ECG Data']
    ECG_data = ECG_data_pd.values

    #Segmentation
    ECG_data = ECG_data[:len(ECG_data) // (time_window*fs)*(time_window*fs)]
    ECG_subarrays_data = np.array_split(ECG_data, len(ECG_data) // (time_window*fs))
    ECG_samples = np.vstack(ECG_subarrays_data)

    #Filtering
    fs = 1000
    nyq = 0.5*fs
    order = 5

    high =0.5
    high = high/nyq
    b, a = butter(order, high, btype = 'high')
    ECG_samples_h = lfilter(b,a,ECG_samples)

    low = 70
    low = low/nyq
    b, a = butter(order, low, btype = 'low')
    ECG_samples_hl = lfilter(b,a,ECG_samples_h)

    notch = 50
    notch = notch/nyq
    b, a = iirnotch(notch, 30)
    ECG_samples_proc = lfilter(b,a,ECG_samples_hl)
    rows = ECG_samples_proc.shape[0]

    #Features extraction
    def r_peaks_detector(i):
      peak_mat = ECG_samples_proc[i,:]
      detectors = Detectors(fs)
      r_peaks_pan = detectors.pan_tompkins_detector(peak_mat)
      r_peaks_pan = np.asarray(r_peaks_pan)
      return r_peaks_pan

    hrv = HRV(fs)

    r_peaks_list = [[]] * rows
    mean_rr_interval_list = [[]] * rows
    HR_list = [[]] * rows
    NN20_list = [[]] * rows
    NN50_list = [[]] * rows
    RMSSD_list = [[]] * rows
    SDSD_list = [[]] * rows
    R_mean_amp_list = [[]] * rows
    HR_biosppy = [[]] * rows
    sdnn_biosppy = [[]] * rows
    rmssd_biosppy = [[]] * rows
    mean_qrs_duration = [[]] * rows
    mean_p_wave_amplitude = [[]] * rows
    mean_t_wave_amplitude = [[]] * rows

    for i in range(rows):
        out = ecg.ecg(signal=ECG_samples_proc[i], sampling_rate=1000, show=False)
        r_peaks = out['rpeaks']
        rr_intervals = np.diff(r_peaks)
        templates = out['templates']
        rpeaks_amplitude = ECG_samples_proc[i][r_peaks]

        r_peaks_list[i] = r_peaks
        mean_rr_interval_list[i] = np.mean(rr_intervals)
        HR_list[i] = np.mean(60 / rr_intervals * 1000)
        NN20_list[i] = hrv.NN20(r_peaks)
        NN50_list[i] = hrv.NN50(r_peaks)
        RMSSD_list[i] = np.mean(hrv.RMSSD(r_peaks))
        SDSD_list[i] = np.mean(hrv.SDSD(r_peaks))
        R_mean_amp_list[i] = np.mean(rpeaks_amplitude)
        HR_biosppy[i] = (60 / np.mean(rr_intervals)) * 1000
        sdnn_biosppy[i] = np.std(rr_intervals)
        rmssd_biosppy[i] = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        mean_qrs_duration[i] = np.mean(np.diff(r_peaks))
        mean_p_wave_amplitude[i] = np.mean(np.abs(np.min(templates, axis=1)))
        mean_t_wave_amplitude[i] = np.mean(np.abs(np.max(templates, axis=1)))


###################################################################################################################################################################################

    EMG_data_pd = data_person['EMG Data']
    EMG_data = EMG_data_pd.values

    #Segmentation
    EMG_data = EMG_data[:len(EMG_data) // (time_window*fs)*(time_window*fs)]
    EMG_subarrays_data = np.array_split(EMG_data, len(EMG_data) // (time_window*fs))
    EMG_samples = np.vstack(EMG_subarrays_data)

    fs = 1000
    nyq = 0.5*fs
    order = 5

    high = 0.5
    high = high/nyq
    b, a = butter(order, high, btype = 'high')
    EMG_samples_h = lfilter(b,a,EMG_samples)

    low =70
    low = low/nyq
    b, a = butter(order, low, btype = 'low')
    EMG_samples_hl = lfilter(b,a,EMG_samples_h)

    notch=50
    notch = notch/nyq
    b, a = iirnotch(notch, 30)
    EMG_samples_proc = lfilter(b,a,EMG_samples_hl)
    rows = EMG_samples_proc.shape[0]

    EMG_mean_list = [[]] * rows
    EMG_std_list = [[]] * rows
    EMG_median_list = [[]] * rows
    EMG_skew_list = [[]] * rows
    EMG_kurt_list = [[]] * rows
    EMG_rms_list = [[]] * rows
    EMG_zcr_list = [[]] * rows
    EMG_ssc_list = [[]] * rows
    EMG_wavelen_list = [[]] * rows
    emg_max_list = [[]] * rows
    emg_min_list = [[]] * rows
    emg_range_list = [[]] * rows
    emg_power_list = [[]] * rows
    emg_crest_factor_list = [[]] * rows
    emg_variance_list = [[]] * rows
    emg_peak_to_peak_list = [[]] * rows
    emg_crest_list = [[]] * rows

    for i in range(rows):
        EMG_mean_list[i] = np.mean(EMG_samples_proc[i])
        EMG_std_list[i] = np.std(EMG_samples_proc[i])
        EMG_median_list[i] = np.median(EMG_samples_proc[i])
        EMG_skew_list[i] = stats.skew(EMG_samples_proc[i])
        EMG_kurt_list[i] = stats.kurtosis(EMG_samples_proc[i])
        EMG_rms_list[i] = np.sqrt(np.mean(np.square(EMG_samples_proc[i])))
        EMG_zcr_list[i] = np.sum(np.diff(np.sign(EMG_samples_proc[i])) != 0) / len(EMG_samples_proc[i])
        EMG_ssc_list[i] = np.sum(np.diff(np.sign(np.diff(EMG_samples_proc[i]))) != 0)
        EMG_wavelen_list[i] = np.sum(np.abs(np.diff(EMG_samples_proc[i])))
        emg_max_list[i] = np.max(EMG_samples_proc[i])
        emg_min_list[i] = np.min(EMG_samples_proc[i])
        emg_range_list[i] = np.max(EMG_samples_proc[i]) - np.min(EMG_samples_proc[i])
        emg_power_list[i] = np.sum(np.square(EMG_samples_proc[i])) / len(EMG_samples_proc[i])
        emg_crest_factor_list[i] = np.max(np.abs(EMG_samples_proc[i])) / np.sqrt(np.mean(np.square(EMG_samples_proc[i])))
        emg_variance_list[i] = np.var(EMG_samples_proc[i])
        emg_peak_to_peak_list[i] = np.max(EMG_samples_proc[i]) - np.min(EMG_samples_proc[i])
        emg_crest_list[i] = np.max(np.abs(EMG_samples_proc[i]))

####################################################################################################################################################################################

    EDA_data_pd = data_person['GSR Data']
    EDA_data = EDA_data_pd.values

    #Segmentation
    EDA_data = EDA_data[:len(EDA_data) // (time_window*fs)*(time_window*fs)]
    EDA_subarrays_data = np.array_split(EDA_data, len(EDA_data) // (time_window*fs))
    EDA_samples = np.vstack(EDA_subarrays_data)

    fs = 1000
    nyq = 0.5*fs
    order = 5

    high = 0.5
    high = high/nyq
    b, a = butter(order, high, btype = 'high')
    EDA_samples_h = lfilter(b,a,EDA_samples)

    low = 70
    low = low/nyq
    b, a = butter(order, low, btype = 'low')
    EDA_samples_hl = lfilter(b,a,EDA_samples_h)

    notch = 50
    notch = notch/nyq
    b, a = iirnotch(notch, 30)
    EDA_samples_proc = lfilter(b,a,EDA_samples_hl)
    rows = EDA_samples_proc.shape[0]

    order = 5
    freqs = [0.05]
    sos = scipy.signal.butter(5, freqs, 'high', output='sos', fs=100)
    phasic = scipy.signal.sosfiltfilt(sos, EDA_samples_proc)

    order = 5
    freqs = [0.05]
    sos = scipy.signal.butter(5, freqs, 'low', output='sos', fs=100)
    tonic = scipy.signal.sosfiltfilt(sos, EDA_samples_proc)

    def scr_peaks(i):
        Eda_dataset= phasic[i, :]
        scr_peaks, _ = find_peaks(Eda_dataset, height=0.1)
        scr_peaks = np.asarray(scr_peaks)
        return scr_peaks

    scr_peaks_list = [[]] * rows
    scl_list = [[]] * rows
    scr_frequency_list = [[]] * rows
    mean_inter_scr_interval_list = [[]] * rows
    mean_scr_amplitude_list = [[]] * rows
    max_scr_amplitude_list = [[]] * rows
    median_scr_amplitude_list = [[]] * rows
    num_scrs_list = [[]] * rows
    mean_scr_duration_list = [[]] * rows
    auc_list = [[]] * rows
    scr_rise_time_list = [[]] * rows
    scr_recovery_time_list = [[]] * rows
    scr_half_recovery_time_list = [[]] * rows
    scr_sensitivity_list = [[]] * rows
    scr_frequency_band_list = [[]] * rows
    scr_energy_list = [[]] * rows
    scr_slope_list = [[]] * rows
    scr_amplitude_variation_list = [[]] * rows
    scr_area_ratio_list = [[]] * rows
    scr_frequency_slope_list = [[]] * rows
    scr_time_ratio_list = [[]] * rows
    scr_tonic_ratio_list = [[]] * rows
    scr_area_peaks_list = [[]] * rows

    for i in range(rows):
        scr_peaks_list[i] = scr_peaks(i)
        scl_list[i] = np.mean(compute_scl(phasic[i, :]))
        scr_frequency_list[i] = np.mean(compute_scr_frequency(phasic[i, :], fs))
        mean_inter_scr_interval_list[i] = compute_inter_scr_interval(phasic[i, :], fs)
        mean_scr_amplitude_list[i] = compute_mean_scr_amplitude(phasic[i, :])
        max_scr_amplitude_list[i] = np.mean(compute_max_scr_amplitude(phasic[i, :]))
        median_scr_amplitude_list[i] = np.mean(compute_median_scr_amplitude(phasic[i, :]))
        num_scrs_list[i] = compute_num_scrs(phasic[i, :])
        mean_scr_duration_list[i] = compute_mean_scr_duration(phasic[i, :], fs)
        auc_list[i] = compute_auc(phasic[i, :])
        scr_rise_time_list[i] = compute_scr_rise_time(phasic[i, :], fs)
        scr_recovery_time_list[i] = compute_scr_recovery_time(phasic[i, :], fs)
        scr_half_recovery_time_list[i] = compute_scr_half_recovery_time(phasic[i, :], fs)
        scr_sensitivity_list[i] = compute_scr_sensitivity(phasic[i, :], tonic[i, :])
        scr_frequency_band_list[i] = compute_scr_frequency_band(phasic[i, :], fs)
        scr_energy_list[i] = compute_scr_energy(phasic[i, :])
        scr_slope_list[i] = compute_scr_slope(phasic[i, :], fs)
        scr_amplitude_variation_list[i] = compute_scr_amplitude_variation(phasic[i, :])
        scr_area_ratio_list[i] = compute_scr_area_ratio(phasic[i, :], tonic[i, :])
        scr_frequency_slope_list[i] = compute_scr_frequency_slope(phasic[i, :], fs)
        scr_time_ratio_list[i] = compute_scr_time_ratio(phasic[i, :], tonic[i, :], fs)
        scr_tonic_ratio_list[i] = compute_scr_tonic_ratio(phasic[i, :], tonic[i, :])
        scr_area_peaks_list[i] = compute_scr_area_peaks(phasic[i, :])

##############################################################################################################################################################################

    feature_lists = {
      'HRV': mean_rr_interval_list,
      'HR': HR_list,
      'NN20': NN20_list,
      'NN50': NN50_list,
      'RMSSD': RMSSD_list,
      'SDSD': SDSD_list,
      'meanRamp': R_mean_amp_list,
      'HRbio': HR_biosppy,
      'sdnnbio': sdnn_biosppy,
      'rmssdbio': rmssd_biosppy,
      'qrs': mean_qrs_duration,
      'pwave': mean_p_wave_amplitude,
      'twave': mean_t_wave_amplitude,
      'EMG_mean': EMG_mean_list,
      'EMG_std': EMG_std_list,
      'EMG_median': EMG_median_list,
      'EMG_skew': EMG_skew_list,
      'EMG_kurt': EMG_kurt_list,
      'EMG_rms': EMG_rms_list,
      'EMG_zcr': EMG_zcr_list,
      'EMG_ssc': EMG_ssc_list,
      'EMG_wavelen': EMG_wavelen_list,
      'emg_max': emg_max_list,
      'emg_min': emg_min_list,
      'emg_range': emg_range_list,
      'emg_power': emg_power_list,
      'emg_crest_factor': emg_crest_factor_list,
      'emg_variance': emg_variance_list,
      'emg_peak_to_peak': emg_peak_to_peak_list,
      'emg_crest': emg_crest_list,
      'scl': scl_list,
      'scr_frequency': scr_frequency_list,
      'mean_inter_scr_interval': mean_inter_scr_interval_list,
      'mean_scr_amplitude': mean_scr_amplitude_list,
      'max_scr_amplitude': max_scr_amplitude_list,
      'median_scr_amplitude': median_scr_amplitude_list,
      'num_scrs': num_scrs_list,
      'mean_scr_duration': mean_scr_duration_list,
      'auc': auc_list,
      'scr_rise_time': scr_rise_time_list,
      'scr_recovery_time': scr_recovery_time_list,
      # 'scr_half_recovery_time': scr_half_recovery_time_list,
      'scr_sensitivity': scr_sensitivity_list,
      # 'scr_frequency_band': scr_frequency_band_list,
      'scr_energy': scr_energy_list,
      'scr_slope': scr_slope_list,
      'scr_amplitude_variation': scr_amplitude_variation_list,
      'scr_area_ratio': scr_area_ratio_list,
      'scr_frequency_slope': scr_frequency_slope_list,
      'scr_time_ratio': scr_time_ratio_list,
      'scr_tonic_ratio': scr_tonic_ratio_list,
      'scr_area_peaks': scr_area_peaks_list
    }

    matrix_dataa = zip(*feature_lists.values())

    data_matrix = np.array(list(matrix_dataa))

    return data_matrix
#%%
time_window = 45  # in seconds

output_ar = processing(time_window)
output = pd.DataFrame(output_ar)


output.columns = ['HRV', 'HR', 'NN20', 'NN50', 'RMSSD', 'SDSD', 'meanRamp', 'HRbio', 'sdnnbio', 'rmssdbio',
                  'qrs', 'pwave', 'twave', 'EMG_mean', 'EMG_std', 'EMG_median', 'EMG_skew', 'EMG_kurt', 'EMG_rms',
                  'EMG_zcr', 'EMG_ssc', 'EMG_wavelen', 'emg_max', 'emg_min', 'emg_range', 'emg_power',
                  'emg_crest_factor', 'emg_variance', 'emg_peak_to_peak', 'emg_crest', 'scl', 'scr_frequency',
                  'mean_inter_scr_interval', 'mean_scr_amplitude', 'max_scr_amplitude', 'median_scr_amplitude',
                  'num_scrs', 'mean_scr_duration', 'auc', 'scr_rise_time', 'scr_recovery_time', 'scr_sensitivity',
                  'scr_energy', 'scr_slope', 'scr_amplitude_variation', 'scr_area_ratio', 'scr_frequency_slope',
                  'scr_time_ratio', 'scr_tonic_ratio', 'scr_area_peaks']
#%%
output.to_csv('C:/Users/tjges/OneDrive/Documents/EPO4/Exposure_50_features/changed_sampling/P2_no_header.csv', index=False, header=False)
# %%
