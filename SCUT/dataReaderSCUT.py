import os
import numpy as np
import tensorflow as tf
import skimage.io as skio
import skimage.transform as sktr

def dataset_iterator(data_path,
                     num_gestures,
                     num_subjects,
                     num_sessions,
                     num_instances,
                     test_set
                     ):
    
    """
    Function to fetch gesture seqeunce paths

    INPUTS:-
    1) data_path: The directory path
    2) num_gestures: The total number of gestures 
    3) num_subjects: The total number of subjects 
    4) num_sessions: The total number of sessions (starting from 1)
    5) num_instances: The total number of instances
    6) test_set: True if test set

    OUTPUTS:-
    1) X: List of lists containing frame paths per gesture sample
    2) y: Corresponding labels
    """
    
    ##### Defining essentials
    X = []
    y = []
    y_id = []

    if(test_set == False):

        for gesture_id in range(num_gestures): # Iterating over Gestures
            print('++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('Gesture ID: '+str(gesture_id)+' Processing')
            
            for subject_id in range(num_subjects): # Iterating over Subjects
                print('-------------------------------------------------')
                print('Subject Processing: '+str(subject_id)+' Processing')

                for instance_id in range(num_instances): # Iterating over Instances
                                 
                    folder_name = data_path + '1_1_'+str(subject_id)+'_'+str(gesture_id)+'_'+str(instance_id) # Name of the folder
                    gesture_curr = [] # List to store processed frames of the current gestures
                    
                    #### Data-Storing
                    ### Gesture Generation
                    for frame_idx, frame_curr in enumerate(sorted(os.listdir(folder_name))): # Iteration over the folder
                        frame_path = folder_name+'/'+frame_curr
                        gesture_curr.append(frame_path)

                    X.append(gesture_curr)
                    y.append(gesture_id)
                    y_id.append(subject_id)

        return X, y, y_id

    if(test_set == True):
                
        for gesture_id in range(num_gestures): # Iterating over Gestures
            print('++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('Gesture ID: '+str(gesture_id)+' Processing')
            
            for subject_id in range(num_subjects): # Iterating over Subjects
                print('-------------------------------------------------')
                print('Subject Processing: '+str(subject_id)+' Processing')

                for instance_id in range(num_instances): # Iterating over Instances

                    for session_id in range(1,num_sessions+1): # Itetating over Sessions

                        folder_name = data_path + '2_'+str(session_id)+'_'+str(subject_id)+'_'+str(gesture_id)+'_'+str(instance_id) # Name of the folder

                        if(os.path.isdir(folder_name)): 

                            gesture_curr = [] # List to store processed frames of the current gestures
                    
                            #### Data-Storing
                            ### Gesture Generation
                            for frame_idx, frame_curr in enumerate(sorted(os.listdir(folder_name))): # Iteration over the folder
                                frame_path = folder_name+'/'+frame_curr
                                gesture_curr.append(frame_path)

                            X.append(gesture_curr)
                            y.append(gesture_id)
                            y_id.append(subject_id)

                        else:
                            pass

        return X, y, y_id
    
class scutReader(tf.keras.utils.Sequence):

    """
    Class to generate data generator for SCUT dataset
    """

    def __init__(self,dims,batch_size,X,y):
        self.dims = dims # (T,H,W,C)
        self.batch_size = batch_size # Required batch size
        self.X = X # List of gesture frame paths
        self.y = y # Corresponding labels
        self.indexes = list(np.arange(len(self.X))) # List comprising index of total entries

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))
    
    def preprocess_frame(self,img_path):

        """
        Function to Read the Frame and Preprocess them.

        INPUTS:-
        1) img_path: Path to the image

        OUTPUTS:-
        1) img_op: Preprocessed image of Dimensions - (64*64*3)
        """
        img_op = sktr.resize(skio.imread(img_path,as_gray=False),(64,64,3))
        return img_op
    
    def __getitem__(self, index):

        """
        Fetching a batch

        INPUTS:-
        1) index: The batch number to be fetched

        OUTPUTS:-
        1) X: Batch of shape (N,T,H,W,C)
        2) y: Labels of shape (N,)
        """
        
        ##### Defining essentials
        X = np.empty((self.batch_size,self.dims[0],self.dims[1],self.dims[2],self.dims[3]))
        y = np.empty((self.batch_size), dtype=int)
        idx_curr = self.indexes[self.batch_size*index:(self.batch_size)*(1+index)] # Current entries

        ##### Populating entries
        for idx, idx_val in enumerate(idx_curr):

            X_curr = []
            X_curr_paths = self.X[idx_val]

            for item in X_curr_paths:
                X_curr.append(self.preprocess_frame(item))
            X_curr = np.array(X_curr)

            X[idx] = X_curr
            y[idx] = self.y[idx_val]

        return X, y

if __name__ == "__main__":
    directory = './Datasets/SCUT-DHGA/SCUT-DHG-Auth/SCUT-DHGA/color_hand/' 
    
    X_train, y_train, y_train_id = dataset_iterator(directory,
                                        6,
                                        143,
                                        1,
                                        10,
                                        False)
    np.savez_compressed('./Datasets/SCUT-DHGA/GBQA/y_train_GBQA-CU-5050_SCUT.npz',y_train)
    np.savez_compressed('./Datasets/SCUT-DHGA/GBQA/y_train_id_GBQA-CU-5050_SCUT.npz',y_train_id)
    #print(X_train,y_train)
    #print(len(X_train),len(y_train))

    X_dev, y_dev, y_dev_id = dataset_iterator(directory,
                                        6,
                                        50,
                                        2,
                                        10,
                                        True)
    np.savez_compressed('./Datasets/SCUT-DHGA/GBQA/y_dev_GBQA-CU-5050_SCUT.npz',y_dev)
    np.savez_compressed('./Datasets/SCUT-DHGA/GBQA/y_dev_id_GBQA-CU-5050_SCUT.npz',y_dev_id)
    #print(X_dev,y_dev)
    #print(len(X_dev),len(y_dev))