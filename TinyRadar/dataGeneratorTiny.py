####### Loading Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

####### Defining Essentials
def dataGenerator(prune):

    ###### Defining Variables and Constants
    X_train = []
    X_dev = []
    y_train = []
    y_dev = []
    y_train_id = []
    y_dev_id = []

    dataset_path = './Datasets/TinyRadar/data_feat'

    frame_len = 5 # Number of Frames in the Input
    doppler_size = 32
    range_size = 492
    total_instances = 105

    train_people = ['0_1','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
    test_people = ['16','17','18','19','20','21','22','23','24','25']
    people = ['0_1','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']

    if(prune == True):
        gestures = ["PinchIndex", "PalmTilt", "FastSwipeRL", "Push", "FingerRub", "Circle"]
    else:
        gestures = ["PinchIndex", "PalmTilt", "FingerSlider", "PinchPinky", "SlowSwipeRL", "FastSwipeRL", "Push", "Pull", "FingerRub", "Circle", "PalmHold"]

    ####### Iterating over the Dataset
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for gdx, gesture_name in enumerate(gestures): # Iterating over different gestures

        for pdx, person in enumerate(people): # Iterating over different users

            path = dataset_path+"/"+"p"+person+"/"+gesture_name+"_"+"1s_"+"wl32_doppl.npy" # Defining Gesture feature file's path
            Gesture_Sequences = np.load(path) # Current Gesture
            Gesture_Sequences = shuffle(Gesture_Sequences) # Shuffling the gesture instances to get a random

            g_count = 0 # For Counting Number of appended gestures
            for gesture_instance in Gesture_Sequences:

                ##### Training Storage
                if(person in train_people): # Considering 60% of the data of a person in training 

                    X_train.append(gesture_instance)
                    y_train.append(gdx)
                    y_train_id.append(pdx)
                    g_count = g_count+1

                ##### Testing Storage
                else: # Considering the remaining 40% of the data of a person in testing
                    X_dev.append(gesture_instance)
                    y_dev.append(gdx)
                    y_dev_id.append(pdx)
                    g_count = g_count+1

            print('Gesture '+str(gesture_name)+' done for person: '+person)
        
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    ######## Creating the Dataset

    ##### Shuffling
    X_train, y_train, y_train_id = shuffle(X_train,y_train,y_train_id)
    X_dev, y_dev, y_dev_id = shuffle(X_dev,y_dev,y_dev_id)

    ##### Data Saving
    if(prune==False):
        np.savez_compressed('./Datasets/TinyRadar/GBQA/X_train_GBQA-CU-5050_Tiny.npz',X_train)
        np.savez_compressed('./Datasets/TinyRadar/GBQA/y_train_GBQA-CU-5050_Tiny.npz',y_train)
        np.savez_compressed('./Datasets/TinyRadar/GBQA/y_train_id_GBQA-CU-5050_Tiny.npz',y_train_id)
        np.savez_compressed('./Datasets/TinyRadar/GBQA/X_dev_GBQA-CU-5050_Tiny.npz',X_dev)
        np.savez_compressed('./Datasets/TinyRadar/GBQA/y_dev_GBQA-CU-5050_Tiny.npz',y_dev) 
        np.savez_compressed('./Datasets/TinyRadar/GBQA/y_dev_id_GBQA-CU-5050_Tiny.npz',y_dev_id)
    else:
        np.savez_compressed('./Datasets/TinyRadar/GBQA/X_train_GBQA-CU-5050-Prune_Tiny.npz',X_train)
        np.savez_compressed('./Datasets/TinyRadar/GBQA/y_train_GBQA-CU-5050-Prune_Tiny.npz',y_train)
        np.savez_compressed('./Datasets/TinyRadar/GBQA/y_train_id_GBQA-CU-5050-Prune_Tiny.npz',y_train_id)
        np.savez_compressed('./Datasets/TinyRadar/GBQA/X_dev_GBQA-CU-5050-Prune_Tiny.npz',X_dev)
        np.savez_compressed('./Datasets/TinyRadar/GBQA/y_dev_GBQA-CU-5050-Prune_Tiny.npz',y_dev) 
        np.savez_compressed('./Datasets/TinyRadar/GBQA/y_dev_id_GBQA-CU-5050-Prune_Tiny.npz',y_dev_id)

    ### Inferring the Shape
    print(np.array(X_train).shape)
    print(np.array(X_dev).shape)
    print(np.array(y_train).shape)
    print(np.array(y_dev).shape)
    print(np.array(y_train_id).shape)
    print(np.array(y_dev_id).shape)

if __name__ == "__main__":
    dataGenerator(True)
    dataGenerator(False)