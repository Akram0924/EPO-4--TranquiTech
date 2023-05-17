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
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2,shuffle=True, random_state=42)

#standard normalize the data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#%%

#Running PCA to plot two features 
# train pca
pca = PCA(n_components=2) # doing pca and keeping only 5 components
pca = pca.fit(X_train)

# perform pca on features
X_train_pca=pca.transform(X_train);
X_test_pca=pca.transform(X_test)

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
    hidden_layer_1_nodes = 20
    hidden_layer_2_nodes = 10
    output_layer = 1

    # initializing a sequential model
    full_model = Sequential()

    # adding layers
    full_model.add(Dense(hidden_layer_1_nodes,input_dim=input_nodes , activation='relu'))
    full_model.add(Dropout(0.1))
    full_model.add(Dense(hidden_layer_2_nodes, activation='relu'))
    full_model.add(Dropout(0.1))
    full_model.add(Dense(output_layer, activation='sigmoid'))
    full_model.summary()

    # Compiling the ANN
    full_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = full_model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=20, batch_size=8, verbose=2)   
    fig, ax = plt.subplots(1)

    ax.plot(history.history['loss'], label = "loss")
    ax.plot(history.history['val_loss'], label = "val_loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch [n]")
    ax.legend()
    return np.rint(full_model.predict(X_test))
predictions = run_nn(X_train,y_train,X_test,y_test)
#%%
print(predictions)

# %%
