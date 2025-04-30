######## Importing libraries
import os                                                                                                         
import gc
import math
import argparse
import numpy as np
from sklearn.preprocessing import normalize as norm
from utils.DGBQA_Score import gbqa_delta_dist_compute

####### Model Arguments and Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name",
                    type=str,
                    help="Name of the Experiment being run, will be used saving the model and correponding outputs")

args = parser.parse_args()

####### Score estimation

##### Defining Essentials
gesture_list = ['Pinch index','Palm tilt','Fast Swipe','Push','Finger rub','Circle']
num_subjects = 5
num_gestures = 6
dgbqa_score = []
Test_Embeddings = np.load('./Embeddings/'+str(args.exp_name)+'.npz')['arr_0']
y_dev = np.load('./Datasets/y_dev_GBQA-CU-5050_Soli.npz')['arr_0']
y_dev_id = np.load('./Datasets/y_dev_id_GBQA-CU-5050_Soli.npz')['arr_0']
y_dev_id = y_dev_id - 5 # Normalizing the IDs from 5-9 to 0-4

##### DGBQA Score
for g_id, gesture_curr in enumerate(gesture_list):
    print('==============================================')
    dgbqa_score_curr, d_c_star_curr, d_cs_curr, dgbqa_score_wo_curr = gbqa_delta_dist_compute(Test_Embeddings,g_id,num_subjects,y_dev,y_dev_id)
    dgbqa_score.append(dgbqa_score_curr)
    print('d_UNQ for '+str(gesture_curr)+' = '+str(d_c_star_curr))  
    print('d_VRB for '+str(gesture_curr)+' = '+str(d_cs_curr)) 