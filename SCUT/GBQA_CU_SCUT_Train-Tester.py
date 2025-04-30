####### Importing Libraries
import os
import argparse
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from dataReaderSCUT import *
from tdsNet import *
plt.switch_backend('agg')

####### Model Arguments and Hyperparameters
parser = argparse.ArgumentParser()

parser.add_argument('--exp_name',
                    type=str,
                    help="Experiment name")

args = parser.parse_args()


####### Dataset generator
directory = './Datasets/SCUT-DHGA/SCUT-DHG-Auth/SCUT-DHGA/color_hand/' 

X_train_list, y_train_list, = dataset_iterator(directory,
                                    6,
                                    143,
                                    1,
                                    10,
                                    False)
train_gen = scutReader([64,64,64,3],24,X_train_list,y_train_list)

####### Model Training

####### Defining Layers and Model

###### Defining Layers
 
##### Input Shapes
T = 64
H = 64
W = 64
C_rdi = 3
M = T-1

##### Convolutional Layers
#### Residual Backbone
### Block-1
conv1_up = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),
                                  activation='relu',padding='same')

### Block-2
res_block21 = ResBlock(64,3)
res_block22 = ResBlock(64,3)

### Block-3
conv3_up = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),
                                  activation='relu',padding='same')
res_block31 = ResBlock(128,3)

### Block-4
conv4_up = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),
                                  activation='relu',padding='same')
res_block41 = ResBlock(256,3)

### Block-5
conv5_up = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),
                                  activation='relu',padding='same')
res_block51 = ResBlock(512,3)

#### Symbiotic Backbone
conv_symb1 = tf.keras.layers.Conv2D(filters=M,kernel_size=(3,3),
                                  activation='relu',padding='same')
conv_symb2 = tf.keras.layers.Conv2D(filters=2*M,kernel_size=(3,3),
                                  activation='relu',padding='same')
conv_symb3 = tf.keras.layers.Conv2D(filters=3*M,kernel_size=(3,3),
                                  activation='relu',padding='same')
conv_symb4 = tf.keras.layers.Conv2D(filters=4*M,kernel_size=(3,3),
                                  activation='relu',padding='same')
conv_symb5 = tf.keras.layers.Conv2D(filters=5*M,kernel_size=(3,3),
                                  activation='relu',padding='same')

#### ISCA-Layers
isca_1 = ISCA(T)
isca_2 = ISCA(T)
isca_3 = ISCA(T)
isca_4 = ISCA(T)
isca_5 = ISCA(T)

#### BE-Fusion Module
be_fusion = BE_Fusion(1)

###### Defining Model

##### Input Layer
Input_Layer = tf.keras.layers.Input(shape=(T,H,W,C_rdi))

##### Convolutional Backbone
#### Block-1
### Symbiotic Backbone
isca_1_op = isca_1(Input_Layer)
conv_symb1_op = conv_symb1(isca_1_op)
### Conv Backbone
conv1_up_op = conv1_up(Input_Layer)

#### Block-2
### Symbiotic Backbone
isca_2_op = isca_2(conv1_up_op)
isca_2_op = tf.keras.layers.Concatenate(axis=-1)([isca_2_op,conv_symb1_op])
conv_symb2_op = conv_symb2(isca_1_op)
### Conv Bacbone
res_block21_op = res_block21(conv1_up_op)
res_block22_op = res_block22(res_block21_op)

#### Block-3
### Symbiotic Backbone
isca_3_op = isca_3(res_block22_op)
isca_3_op = tf.keras.layers.Concatenate(axis=-1)([isca_3_op,conv_symb2_op])
conv_symb3_op = conv_symb3(isca_3_op)
conv_symb3_op = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_symb3_op)
### Conv Backbone
res_block22_op = tf.keras.layers.MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2))(res_block22_op)
conv3_up_op = conv3_up(res_block22_op)
res_block31_op = res_block31(conv3_up_op)

#### Block-4
### Symbiotic Backbone
isca_4_op = isca_4(res_block31_op)
isca_4_op = tf.keras.layers.Concatenate(axis=-1)([isca_4_op,conv_symb3_op])
conv_symb4_op = conv_symb4(isca_4_op)
### Conv Backbone
conv4_up_op = conv4_up(res_block31_op)
res_block41_op = res_block41(conv4_up_op)

#### Block-5
### Symbiotic Backbone
isca_5_op = isca_5(res_block41_op)
isca_5_op = tf.keras.layers.Concatenate(axis=-1)([isca_5_op,conv_symb4_op])
conv_symb5_op = conv_symb5(isca_5_op)
conv_symb5_op = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_symb5_op)
### Conv Backbone
res_block41_op = tf.keras.layers.MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2))(res_block41_op)
conv5_up_op = conv5_up(res_block41_op)
res_block51_op = res_block51(conv5_up_op)

##### Fusion Layers
#### Pooling Layers
b_op = tf.keras.layers.GlobalAveragePooling2D()(conv_symb5_op)
p_op = tf.keras.layers.GlobalAveragePooling3D()(res_block51_op)

#### Dense Layers
b_op = tf.keras.layers.Dense(128,activation='relu')(b_op)
p_op = tf.keras.layers.Dense(128,activation='relu')(p_op)

#### Fusion
be_op = be_fusion(p_op,b_op)

##### Output Layer
dense1 = tf.keras.layers.Dense(32,activation='relu')(be_op)
dense2 = tf.keras.layers.Dense(6,activation='softmax')(dense1)

###### Compiling Model
model = tf.keras.models.Model(inputs=Input_Layer,outputs=dense2)
model.load_weights('./Models/'+args.exp_name+'.h5') 
model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()
#tf.keras.utils.plot_model(model)

###### Testing Model - SoftMax Style               
def normalisation_layer(x):   
    return(tf.math.l2_normalize(x, axis=1, epsilon=1e-12))

predictive_model = tf.keras.models.Model(inputs=model.input,outputs=model.layers[-2].output)
predictive_model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

Input_Layer_rdi = tf.keras.layers.Input((T,H,W,C_rdi))
op_1 = predictive_model(Input_Layer_rdi)
final_norm_op = tf.keras.layers.Lambda(normalisation_layer)(op_1)

testing_model = tf.keras.models.Model(inputs=Input_Layer_rdi,outputs=final_norm_op)
testing_model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

##### Nearest Neighbor Classification
Test_Embeddings = testing_model.predict(train_gen)

col_mean = np.nanmean(Test_Embeddings, axis=0)
inds = np.where(np.isnan(Test_Embeddings))
#print(inds)
Test_Embeddings[inds] = np.take(col_mean, inds[1])

#### Computing and Saving Output 
np.savez_compressed('./Embeddings/'+args.exp_name+'.npz',Test_Embeddings)