####### Importing Libraries
import os
import cv2
import argparse
import numpy as np
from sklearn.utils import shuffle

####### Model Arguments and Hyperparameters
parser = argparse.ArgumentParser()

parser.add_argument("--s_id",
                    default=1,
                    type=str,
                    help='The list testing subject ids')

args = parser.parse_args()

###### Defining Essentials
X_train = []
X_dev = []
y_train = []
y_train_id = []
y_dev = []
y_dev_id = []
total_frames = 60 # 4s of video @ 15 fps. We will extract in total 61 frames but will 
H = 64 # Resizing size of the Height Dimensions
W = 64 # Resizing size of the Width Dimensions
data_folder = './Datasets/HandLogin/Data'

###### Frame Generation
def frame_gen(filepath,frame_num):

    """
    Function to Process Frames

    INPUTS:-
    1) filepath: Path to the Image Folder
    2) frame_num: The Index of the frame

    OUTPUTS:-
    1) op_frame: Output of frame Processing
    """

    ##### Image Path
    filepath_lsb = filepath+'/LSB'+'/LSB'+str(frame_num)+'.png'
    filepath_msb = filepath+'/MSB'+'/MSB'+str(frame_num)+'.png'

    ##### Image Reading
    #### LSB File
    img_lsb = cv2.cvtColor(cv2.imread(filepath_lsb), cv2.COLOR_BGR2GRAY)

    #### MSB File
    ### File-Reading
    img_msb = cv2.imread(filepath_msb)[:,:,0]

    ### File-Conversion
    img_msb_list = [] # List to store updated MSB Image
    
    for i in range(img_msb.shape[0]):

        img_msb_row_curr = [] # List to store values

        for j in range(img_msb.shape[1]):

            img_msb_curr = img_msb[i,j] # Current Image
            img_msb_curr_bin = np.binary_repr(img_msb_curr) # Binary Representation

            for k in range(8): # Number of Indexes to be added

                img_msb_curr_bin = img_msb_curr_bin + '0' # Appending 0s at the end to make it 16 bit

            img_msb_row_curr.append(int(img_msb_curr_bin,2))

        img_msb_list.append(img_msb_row_curr)

    img_msb = np.array(img_msb_list)

    #### LSB-MSB Combination
    op_frame = np.double(img_lsb) + np.double(img_msb)
    op_frame = (op_frame/256).astype('uint8')
    op_frame = op_frame[150:400,200:450] # Cropping the Frame Adequately
    op_frame = cv2.resize(op_frame, (H,W),  interpolation = cv2.INTER_NEAREST_EXACT) # Resizing the Ouptut Frame
    return op_frame

###### Background Substraction
def bg_substractor(frame,bg_frame,threshold):

    """
    Function to Substract Background

    INPUTS:-
    1) frame: Frame from which Background is to be substracted
    2) bg_frame: Background Frame
    3) threshold: Threshold for Foreground Mask

    OUTPUTS:-
    1) op_frame: Output of frame Processing
    """

    fg_mask = ((bg_frame - frame) > threshold) # Foreground Mask
    op_frame = np.multiply(frame,fg_mask) # Mask Application
    return op_frame

###### Optical Flow Estimation
def optical_flow_estimator(video_seq):

    """
    Estimating Franeback Flow for Video Sequence

    INPUTS:-
    1) video_seq: Input video sequence of shape (T,H,W) in this case

    OUTPUTS:-
    1) flow_op: Output Flow of the shape (T,H,W,2). 
                Here two channels correspond to Magnitude and Direction of the Flow
    """

    ##### Defining Essentials
    flow_op = [] # List to store Flow Outputs
    frames_total = int(video_seq.shape[0]) # A Count on total number of flows in the Video Sequence

    ##### Iterating till Tth from the first Frame
    for frame_idx in range(1,frames_total):

        frame_curr = video_seq[frame_idx] # Current Frame
        frame_prev = video_seq[frame_idx-1] # Previous Frame

        flow = cv2.calcOpticalFlowFarneback(frame_prev, frame_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0) # Optical Flow Extraction
        mag, dir = cv2.cartToPolar(flow[...,0],flow[...,1]) # Magnitude and Direction of Flow
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) # Scaling the Magnitude
        dir = dir*180/np.pi/2 # Scaling the Directional Flow
        flow = np.stack([mag,dir],axis=-1) # Stacking the Arrays
        flow_op.append(flow) # Appending the Flow

    return np.array(flow_op)

###### Symboitic Feature Maps
def symbiotic_extractor(video_seq):

    """
    Estimating Franeback Flow for Video Sequence

    INPUTS:-
    1) video_seq: Input video sequence of shape - (T,H,W) in this case

    OUTPUTS:-
    1) symb_op: Symbiotic Feature of shape - (T,H,W). 
    """
    ##### Defining Essentials
    symb_op = [] # List to store Sym Outputs
    frames_total = int(video_seq.shape[0]) # A Count on total number of flows in the Video Sequence

    ##### Iterating till Tth from the First Frame
    for frame_idx in range(1,frames_total):

        frame_curr = video_seq[frame_idx] # Current Frame
        frame_prev = video_seq[frame_idx-1] # Previous Frame
        frame_curr = frame_curr - frame_prev # Symbiotic Feature Extraction
        symb_op.append(frame_curr) # Appending it to the list

    return np.array(symb_op)

###### Gesture Sequence Generator
def gesture_seq_gen(folder_path,threshold):

    """
    Function to return a Gesture Sequence from the Folder Path

    INPUTS:-
    1) folder_path: Path to the Input Folder
    2) threshold: Background substraction threshold0

    OUTPUTS:-
    gesture_seq: Output Sequence of the Gesture
    """

    ##### Defining Essentials
    gesture_seq = [] # List to store frames
    bg_frame = frame_gen(folder_path,1) # Background Frame
    total_frames = len(os.listdir(folder_path+'/LSB')) # Computing Total Number of Frames
    frame_names = os.listdir(folder_path+'/LSB') # List to store the names of all the available frames

    ##### Iterating over the Folder 
    for frame_idx in range(1,total_frames,2): # Dropping the First Frame, stepping@ 15fps instead of 30

        frame_name_curr = 'LSB'+str(frame_idx+1)+'.png' # Fetching the current frame name

        if(frame_name_curr in frame_names): # Checking if the Frame exists or not
            #print(frame_name_curr)
            frame_curr = frame_gen(folder_path,frame_idx+1) # Getting the Current Frame
            #frame_curr = bg_substractor(frame_curr,bg_frame,threshold) # Background Substraction
            gesture_seq.append(frame_curr)

        else:
            break 

    ##### Frame-Rate Adjustments
    total_frames_curr = len(gesture_seq) # Total Number of Frames in the Gesture Sequence

    if(total_frames_curr < 60): # Zero-Padding: If frames are less than 60 

        for f_rem in range(60 - len(gesture_seq)):
            frame_add = np.zeros((H,W),dtype=np.double) # Frame to be Added
            gesture_seq.append(frame_add)

        gesture_seq = np.array(gesture_seq)

    elif(len(gesture_seq) >= 60): # Z

        gesture_seq = np.array(gesture_seq)[:60] # Slicing: If frames are greater than or equal to 60

    return gesture_seq

###### Iterating over Datasets
for s_id, s_folder_name in enumerate(np.sort(os.listdir(data_folder))):

    s_folder = data_folder+'/'+s_folder_name
    print('=========================================')
    print('Processing for Subject: '+str(s_id))

    for g_id, g_folder in enumerate(np.sort(os.listdir(s_folder))):

        g_folder = s_folder+'/'+g_folder
        print('++++++++++++++++++++++++++++++++++++++++++')
        print('Processing for Gesture: '+str(g_id))

        for instance_id, instance in enumerate(np.sort(os.listdir(g_folder))):

            print('Processing for Instance: '+str(instance_id))

            if(s_folder_name != args.s_id): # Saving First 6 Instances as Training Instances

                instance = g_folder + '/' + instance # Extracting instance-folder path
                gesture_seq_op = gesture_seq_gen(instance,5) # Extacting Gesture Video Sequence
                #gesture_seq_op = symbiotic_extractor(gesture_seq_op) # Extracting Symbiotic Features

                X_train.append(gesture_seq_op)
                y_train.append(g_id)
                y_train_id.append(s_id)

            else: # Saving the remaining 4 Instances for Testing

                instance = g_folder + '/' + instance # Extracting instance-folder path
                gesture_seq_op = gesture_seq_gen(instance,5) # Extacting Gesture Video Sequence
                #gesture_seq_op = symbiotic_extractor(gesture_seq_op) # Extracting Symbiotic Features

                X_dev.append(gesture_seq_op)
                y_dev.append(g_id)
                y_dev_id.append(s_id) 

####### Creating the Dataset

###### Shuffling
X_train, y_train, y_train_id = shuffle(np.reshape(X_train,(len(X_train),total_frames,H,W,1)),np.array(y_train),np.array(y_train_id))
X_dev, y_dev, y_dev_id = shuffle(np.reshape(X_dev,(len(X_dev),total_frames,H,W,1)),np.array(y_dev),np.array(y_dev_id))
#X_train, y_train, y_train_id = shuffle(np.array(X_train),np.array(y_train),np.array(y_train_id))
#X_dev, y_dev, y_dev_id = shuffle(np.array(X_dev),np.array(y_dev),np.array(y_dev_id))

##### Data Saving
np.savez_compressed('./Datasets/HandLogin/GBQA/X_train_GBQA-CU-'+args.s_id+'_HandLogin.npz',X_train)
np.savez_compressed('./Datasets/HandLogin/GBQA/y_train_GBQA-CU-'+args.s_id+'_HandLogin.npz',y_train)
np.savez_compressed('./Datasets/HandLogin/GBQA/y_train_id_GBQA-CU-'+args.s_id+'_HandLogin.npz',y_train_id)
np.savez_compressed('./Datasets/HandLogin/GBQA/X_dev_GBQA-CU-'+args.s_id+'_HandLogin.npz',X_dev)
np.savez_compressed('./Datasets/HandLogin/GBQA/y_dev_GBQA-CU-'+args.s_id+'_HandLogin.npz',y_dev) 
np.savez_compressed('./Datasets/HandLogin/GBQA/y_dev_id_GBQA-CU-'+args.s_id+'_HandLogin.npz',y_dev_id)

### Inferring the Shape
print(np.array(X_train).shape)
print(np.array(X_dev).shape)
print(np.array(y_train).shape)
print(np.array(y_dev).shape)
print(np.array(y_train_id).shape)
print(np.array(y_dev_id).shape)