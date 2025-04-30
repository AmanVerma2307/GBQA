####### Importing Libraries
import os
import argparse
import numpy as np
import skimage.io as skio
import skimage.transform as sktr
from sklearn.utils import shuffle

####### Deciphering the Fold index
parser = argparse.ArgumentParser()

parser.add_argument("--fold_id",
                    type=int,
                    help="Testing fold id, total five folds")

args = parser.parse_args()

if(args.fold_id == 1):
    testing_id = list(np.arange(0,28,1))

elif(args.fold_id == 2):
    testing_id = list(np.arange(28,56,1))

elif(args.fold_id == 3):
    testing_id = list(np.arange(56,84,1))

elif(args.fold_id == 4):
    testing_id = list(np.arange(84,112,1))

elif(args.fold_id == 5):
    testing_id = list(np.arange(112,143,1))

####### Iterating over the Directory
###### Defining Essentials
num_gestures = 6
num_subjects = 143
num_sessions = 10
T = 64
H = 64
W = 64
directory = './Datasets/SCUT-DHGA/SCUT-DHG-Auth/SCUT-DHGA/color_hand/' 
X_train = []
y_train = []
y_train_id = []
X_dev = []
y_dev = []
y_dev_id = []

###### Read and Preprocessing Frame
def preprocess_frame(img_path):

    """
    Function to Read the Frame and Preprocess them.

    INPUTS:-
    1) img_path: Path to the image

    OUTPUTS:-
    1) img_op: Preprocessed image of Dimensions - (64*64*3)
    """
    img_op = sktr.resize(skio.imread(img_path,as_gray=False),(64,64,3))
    return img_op

###### Iteration Loop
for gesture_id in range(num_gestures): # Iterating over Gestures
    print('++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Gesture ID: '+str(gesture_id)+' Processing')
    
    for subject_id in range(num_subjects): # Iterating over Subjects
        print('-------------------------------------------------')
        print('Subject Processing: '+str(subject_id)+' Processing')

        for session_id in range(num_sessions): # Iterating over Sessions

            folder_name = directory + '1_1_'+str(subject_id)+'_'+str(gesture_id)+'_'+str(session_id) # Name of the folder
            gesture_curr = [] # List to store processed frames of the current gestures
            
            #### Data-Storing
            ### Gesture Generation
            for frame_idx, frame_curr in enumerate(sorted(os.listdir(folder_name))): # Iteration over the folder
                frame_path = folder_name+'/'+frame_curr
                gesture_curr.append(preprocess_frame(frame_path))

            gesture_curr = np.array(gesture_curr) # Array Formation
            gesture_curr = np.reshape(gesture_curr,(T,H,W,3)) # Reshape Operation
            #gesture_curr = TDN_Generator(gesture_curr) # TDN Map Generation

            ### Storing Operation
            if(subject_id not in testing_id):
                X_train.append(gesture_curr)
                y_train.append(gesture_id)
                y_train_id.append(subject_id)
            else:
                X_dev.append(gesture_curr)
                y_dev.append(gesture_id)
                y_dev_id.append(subject_id)

####### Creating the Dataset

###### Shuffling
X_train, y_train, y_train_id = shuffle(np.array(X_train),np.array(y_train),np.array(y_train_id))
X_dev, y_dev, y_dev_id = shuffle(np.array(X_dev),np.array(y_dev),np.array(y_dev_id))                        

###### SavingShuffled Dataset
np.savez_compressed('./Datasets/SCUT-DHGA/GBQA/X_train_GBQA-CU-'+str(args.fold_id)+'_SCUT.npz',np.array(X_train))
np.savez_compressed('./Datasets/SCUT-DHGA/GBQA/y_train_GBQA-CU-'+str(args.fold_id)+'_SCUT.npz',np.array(y_train))
np.savez_compressed('./Datasets/SCUT-DHGA/GBQA/y_train_id_GBQA-CU-'+str(args.fold_id)+'_SCUT.npz',np.array(y_train_id))
np.savez_compressed('./Datasets/SCUT-DHGA/GBQA/X_dev_GBQA-CU-'+str(args.fold_id)+'_SCUT.npz',np.array(X_dev))
np.savez_compressed('./Datasets/SCUT-DHGA/GBQA/y_dev_GBQA-CU-'+str(args.fold_id)+'_SCUT.npz',np.array(y_dev))
np.savez_compressed('./Datasets/SCUT-DHGA/GBQA/y_dev_id_GBQA-CU-'+str(args.fold_id)+'_SCUT.npz',np.array(y_dev_id))

###### Inferring the Shape
print(np.array(X_train).shape)
print(np.array(X_dev).shape)
print(np.array(y_train).shape)
print(np.array(y_dev).shape)
print(np.array(y_train_id).shape)
print(np.array(y_dev_id).shape)