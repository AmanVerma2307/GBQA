####### Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tensorflow as tf
import os                                                                                                         
import gc
import math
import pydot

####### Loading Dataset
X_train = np.load('./Datasets/X_train_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']
X_dev = np.load('./Datasets/X_dev_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']
y_train = np.load('./Datasets/y_train_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']
y_dev = np.load('./Datasets/y_dev_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']

####### Model Training
####### Defining Layers and Model

###### Defining Layers
 
##### Input Shapes
T = 40
H = 32
W = 32
C_rdi = 4

##### Convolutional Layers

#### Res3DNet
conv11_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
conv12_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
conv13_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
maxpool_1 = tf.keras.layers.MaxPooling3D(pool_size=(1,2,2))

conv21_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')
conv22_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')
conv23_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')

###### Defining Model

##### Input Layer
Input_Layer = tf.keras.layers.Input(shape=(T,H,W,C_rdi))

##### Conv Layers

#### Res3DNet
### Residual Block - 1
conv11_rdi = conv11_rdi(Input_Layer)
conv12_rdi = conv12_rdi(conv11_rdi)
conv13_rdi = conv13_rdi(conv12_rdi)
conv13_rdi = tf.keras.layers.Add()([conv13_rdi,conv11_rdi])
conv13_rdi = maxpool_1(conv13_rdi)

### Residual Block - 2
conv21_rdi = conv21_rdi(conv13_rdi)
conv22_rdi = conv22_rdi(conv21_rdi)
conv23_rdi = conv23_rdi(conv22_rdi)
conv23_rdi = tf.keras.layers.Add()([conv23_rdi,conv21_rdi])

##### Output Layer
gap_op = tf.keras.layers.GlobalAveragePooling3D()(conv23_rdi)
dense1 = tf.keras.layers.Dense(256,activation='relu')(gap_op)
dropout1 = tf.keras.layers.Dropout(rate=0.2)(dense1)

### Softmax Output Layer
dense2 = tf.keras.layers.Dense(256,activation='relu')(dropout1)
dense3 = tf.keras.layers.Dense(6,activation='softmax')(dense2)

###### Compiling Model
model = tf.keras.models.Model(inputs=Input_Layer,outputs=dense3)
model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()
#tf.keras.utils.plot_model(model)

##### Defining Callbacks
filepath= "./Models/GBQA_Res3D_CU-5050_Soli.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,monitor='val_accuracy',save_best_only=True,mode='max')

###### Training the Model
history = model.fit(X_train,y_train,epochs=100,batch_size=32,
                    validation_data=(X_dev,y_dev),validation_batch_size=32,
                   callbacks=checkpoint)

##### Saving Training Metrics
np.save('./Model History/GBQA_CU-5050_Soli.npy',history.history)
 