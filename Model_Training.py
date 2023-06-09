#%%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense,  Dropout
import keras as keras
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from ecgdetectors import Detectors
from hrv import HRV
import pickle
from scipy import signal
from scipy.signal import butter, iirnotch, lfilter
from sklearn import datasets, svm
from sklearn import metrics
import seaborn as sns

#%%

#data_x = 10*np.random.rand(6,20)
#data_y = np.random.randint(2, size=20)[np.newaxis,:]
#data = np.concatenate((data_x,data_y),axis=0)

#%%
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
subject = 'S2'

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
#%%
#Combining stress and baseline sets into one sample set
samples = np.vstack((samples_baseline, samples_stress))

#Combinine label set into one
labels_tot = np.vstack((labels_baseline, labels_stress))
print(samples.shape)

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
print(data.shape)
def load_data(data):
    y = data[-1,:][np.newaxis,:]
    x = data[:-1,:]
    return x, y 

def load_data_T(data):
    y = data[:,-1]
    x = data[:,:-1]
    return x, y 
x, y = load_data(data)
print(x.shape,y.shape)
#%%
#Data must be transposed to fit sklearn train test split
X, y = load_data(data)
def split_and_PCA (X, y):
    X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2,shuffle=True, random_state=42)

    #standard normalize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #Running PCA to plot two features 
    # train pca
    pca = PCA(n_components=2) # doing pca and keeping only 5 components
    pca = pca.fit(X_train)

    # perform pca on features
    X_train_pca=pca.transform(X_train);
    X_test_pca=pca.transform(X_test)
    return(X_train, X_test, y_train, y_test, X_train_pca, X_test_pca)
X_train, X_test, y_train, y_test, X_train_pca, X_test_pca = split_and_PCA(X, y)

def plot_2class(x_in,y_in):
    #Create a figure
    plt.figure()
    colors = ['tab:blue', 'tab:orange']
    #iterate over the classes
    for i in range(2):
        #select only the points with class i and plot them in the right colors
        mask = (y_in.astype(int)==i+1)
        plt.scatter(x_in[0,mask[0,:]], x_in[1,mask[0,:]], marker = 'o', color = colors[i], label = 'y = {}'.format(i+1))
    #finish the plot
    plt.legend()
    plt.axis('scaled')
    plt.xlabel('PCA Feature 1'); plt.ylabel('PCA Feature 2');
   
plot_2class(X_train_pca.T,y_train.T)
# %%
def run_nn(X_train, y_train, X_test, y_test):
    #Creating a simple neural network, sigmoid activation
    input_nodes = X_train.shape[1]
    hidden_layer_1_nodes = 10
    hidden_layer_2_nodes = 20
    hidden_layer_3_nodes = 2
    output_layer = 1

    # initializing a sequential model
    full_model = Sequential()

    # adding layers
    full_model.add(Dense(hidden_layer_1_nodes,input_dim=input_nodes , activation='relu'))
    full_model.add(Dropout(0.1))
    full_model.add(Dense(hidden_layer_2_nodes, activation='relu'))
    full_model.add(Dropout(0.1))
    full_model.add(Dense(hidden_layer_3_nodes, activation='relu'))
    full_model.add(Dropout(0.1))
    full_model.add(Dense(output_layer, activation='sigmoid'))
    full_model.summary()

    # Compiling the ANN
    full_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = full_model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=40, batch_size=10, verbose=2)   
    fig, ax = plt.subplots(1)

    ax.plot(history.history['loss'], label = "loss")
    ax.plot(history.history['val_loss'], label = "validation loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch [n]")
    ax.legend()
    return np.rint(full_model.predict(X_test)), np.rint(full_model.predict(X_train))

predictions_test, predictions_train = run_nn(X_train_pca,y_train,X_test_pca,y_test)
errors_test = np.sum(y_test != predictions_test)
errors_train = np.sum(y_train != predictions_train)

print("misclassifications = ", errors_test)
print("misclassifications = ", errors_train)

# %% Linear Model 
def run_svm(X_train, y_train, X_test, y_test, X_self):
    model = svm.SVC()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    X_self_prediction = model.predict(X_self)
    score = model.score(X_test, y_test)
    cm = metrics.confusion_matrix(y_test, predictions)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 10);

    return predictions, score, X_self_prediction

# %%
def import_recording():
   Fs = 150 #Sampling Frequency of our sensor
   df = pd.read_csv(r'C:/Users/tjges/OneDrive/Documents/EPO4/Recording/laatsteTestDinsdag.csv')
   data = pd.DataFrame(df, columns = ['ECG Data'] ).T
   one_minute = np.array(data)[:,:150*60]
   return one_minute
one_min = import_recording()
#%%
def plot_recording(recording, fs):
    recording = recording[6000:6000+10*fs]
    t=np.arange(0,recording.size*(1/fs),(1/fs))
    t=t[:recording.size]

    plt.figure(figsize=(12,4))
    plt.plot(t,recording/max(recording))
    plt.xlabel('$Time (s)$') 
    plt.ylabel('$ECG$') 
plot_recording(np.ravel(one_min), 150)

def filter_recording(recording, fs):
    nyq = 0.5*fs
    order=5
    t=np.arange(0,recording.size*(1/fs),(1/fs))
    t=t[:recording.size]

    # highpass filter
    high=0.5
    high= high/nyq
    b, a = butter(order, high, btype = 'high')
    ecg_h = lfilter(b,a,recording)

    # lowpass filter
    low=70
    low= low/nyq
    b, a = butter(order, low, btype = 'low')
    ecg_hl = lfilter(b,a,ecg_h)

    # notch filter
    notch=50
    notch = notch/nyq
    b, a = iirnotch(notch, 30)
    ecg_filtered = lfilter(b,a,ecg_hl)[100:]

    plt.figure(figsize=(12,4))
    plt.plot(t,recording/max(recording),label="raw ECG")
    plt.plot(t[100:],ecg_filtered/max(ecg_filtered), label="filtered ECG")
    plt.xlabel('$Time (s)$') 
    plt.ylabel('$ECG$') 
    plt.legend()
    return ecg_filtered
ecg_filtered = filter_recording(np.ravel(one_min), 150)
#%%
def feature_extraction(ecg, fs):
    detectors = Detectors(fs)

    r_peaks_pan = detectors.pan_tompkins_detector(ecg)
    r_peaks_pan = np.asarray(r_peaks_pan)

    hrv_class = HRV(fs)
    feat_nn20=hrv_class.NN20(r_peaks_pan)
    feat_nn50=hrv_class.NN50(r_peaks_pan)
    feat_rmssd=hrv_class.RMSSD(r_peaks_pan)
    features = np.array([feat_nn20, feat_nn50, feat_rmssd, np.mean(r_peaks_pan)])[:,np.newaxis]
    print(features.shape)

    return features.T
X_self = feature_extraction(ecg_filtered, 150)
# %% 
predictions, score, X_self_prediction = run_svm(X_train, y_train, X_test, y_test, X_self)
print("Tijn's Predicted Label =",X_self_prediction)

# %%
