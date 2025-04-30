####### Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
import h5py 
import json
from sklearn import preprocessing
from pathlib import Path
import os                                                                                                         
import gc
import argparse

####### Reading Dataset

###### Modules for Data Reading

##### Data Reader - Range Doppler

def rdi_reader(dir_path):

    """
    Function to read Range-Doppler Image Sequence

    INPUTS:-
    1)dir_path : Path of Input Sequence

    OUTPUTS:-
    1)rdi_seq : Numpy Array of Shape (Frame,Height,Width,Channel)

    """
    use_channel = 0
    with h5py.File(dir_path, 'r') as f:
        # Data and label are numpy arrays
        data1 = f['ch{}'.format(use_channel)][()]
        label = f['label'][()]

    use_channel = 1
    with h5py.File(dir_path, 'r') as f:
        # Data and label are numpy arrays
        data2 = f['ch{}'.format(use_channel)][()]
        label = f['label'][()]

    use_channel = 2
    with h5py.File(dir_path, 'r') as f:
        # Data and label are numpy arrays
        data3 = f['ch{}'.format(use_channel)][()]
        label = f['label'][()]

    use_channel = 3
    with h5py.File(dir_path, 'r') as f:
        # Data and label are numpy arrays
        data4 = f['ch{}'.format(use_channel)][()]
        label = f['label'][()]    

    data = np.dstack((data1,data2,data3,data4))
    data = np.reshape(data,(data.shape[0],32,32,4))

    if (data.shape[0] < 40): # Zero-Padding Smaller Sequence Lengths
        difference = int(40-data.shape[0]) 
        zero_padding = np.zeros((difference,32,32,4))
        rdi_seq = np.concatenate((data,zero_padding),axis=0)
    else:
        rdi_seq = data[:40]

    return rdi_seq

###### Dataset Creation
##### Defining Essentials
main_dir = './SoliData/dsp'
label_dict = {2:0, 3:1, 5:2, 6:3, 8:4, 9:5, 10:6, 11:7, 12:8, 13:9}

##### Extracting the Required items from the main directory

#### Defining Essentials - For Training Dataset
gesture_id = [0,1,2,3,4,5,6,7,8,9,10] # To select gestures for training
max_frame_number = 40
instance_id = [i for i in range(0,25)] #  To select instances for training : First 15 instances for training
session_id = [2,3,5,6,8] # To select the Users (Sessions) for training
 
instance_list_train = []
label_list_train_g_id = [] # List to Store Gesture Index for Training
label_list_train_p_id = [] # List to Store Person Index

for idx, i in enumerate(gesture_id):
    for j in session_id:
        for k in instance_id:
            item = str(i)+'_'+str(j)+'_'+str(k)+'.h5'
            instance_list_train.append(item)
            label_list_train_g_id.append(idx) # For gesture_id along with gesture
            label_list_train_p_id.append(label_dict[j]) # For person_id along with the gesture

#### Defining Essentials - For Testing Dataset
gesture_id = [0,1,2,3,4,5,6,7,8,9,10] # To select gestures for testing 
max_frame_number = 40
instance_id = [i for i in range(0,25)] # To select instance for testing : Last 10 instances for testing
session_id = [9,10,11,12,13] # To select the Users (Sessions) for testing

instance_list_dev = []
label_list_dev_g_id = [] # List to Store Gesture Index for Evaluation
label_list_dev_p_id = [] # List to Store Person Index for Evaluation

for idx, i in enumerate(gesture_id):
    for j in session_id:
        for k in instance_id:
            item = str(i)+'_'+str(j)+'_'+str(k)+'.h5'
            instance_list_dev.append(item)
            label_list_dev_g_id.append(idx) # For gesture_id along with gesture
            label_list_dev_p_id.append(label_dict[j]) # For person_id along with the gesture

##### Numpy Array Creation
X_train = []
X_dev = []
main_folder_path = './SoliData/dsp'

#### Reading and Appending Dataset
### Training Set 
for item in instance_list_train:
    item = os.path.join(main_folder_path,item)
    X_train.append(rdi_reader(item))
X_train = np.array(X_train)
y_train = np.array(label_list_train_g_id)
y_train_id = np.array(label_list_train_p_id)

X_train, y_train, y_train_id = shuffle(X_train, y_train, y_train_id)

### Test Set
for item in instance_list_dev:
    item = os.path.join(main_folder_path,item)
    X_dev.append(rdi_reader(item))
X_dev = np.array(X_dev)
y_dev = np.array(label_list_dev_g_id)
y_dev_id = np.array(label_list_dev_p_id)

X_dev, y_dev, y_dev_id = shuffle(X_dev, y_dev, y_dev_id)

print(X_train.shape)
print(X_dev.shape)
print(y_train.shape)
print(y_dev.shape)
print(y_train_id.shape)
print(y_dev_id.shape)

#### Saving the Dataset
np.savez_compressed('./Datasets/X_train_GBQA-CU-5050-Full_Soli.npz',X_train)
np.savez_compressed('./Datasets/y_train_GBQA-CU-5050-Full_Soli.npz',y_train)
np.savez_compressed('./Datasets/y_train_id_GBQA-CU-5050-Full_Soli.npz',y_train_id)
np.savez_compressed('./Datasets/X_dev_GBQA-CU-5050-Full_Soli.npz',X_dev)
np.savez_compressed('./Datasets/y_dev_GBQA-CU-5050-Full_Soli.npz',y_dev) 
np.savez_compressed('./Datasets/y_dev_id_GBQA-CU-5050-Full_Soli.npz',y_dev_id)
