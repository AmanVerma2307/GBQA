####### Importing Libraries
import numpy as np
import tensorflow as tf
from utils.msConv3D import msConv3D 
from utils.ViViT import Tubelet_Embedding, PositionEmbedding, Encoder

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
num_layers = 2
d_model = 32
num_heads = 16
dff_dim = 128
p_t = 5
p_h = 5
p_w = 5
n_t = (((T - p_t)//p_t)+1)
n_h = (((H - p_h)//p_h)+1)
n_w = (((W - p_w)//p_w)+1)
max_seq_len = n_t*n_h*n_w
pe_input = n_t*n_h*n_w
rate = 0.3
batch_size = 32

##### Convolutional Layers

#### Res3DNet
conv11_rdi = msConv3D(16)
conv12_rdi = msConv3D(16)
conv13_rdi = msConv3D(16)
maxpool_1 = tf.keras.layers.MaxPooling3D(pool_size=(1,2,2))

conv21_rdi = msConv3D(32)
conv22_rdi = msConv3D(32)
conv23_rdi = msConv3D(32)

##### ViViT
tubelet_embedding_layer = Tubelet_Embedding(d_model,(p_t,p_h,p_w))
positional_embedding_encoder = PositionEmbedding(max_seq_len,d_model)
enc_block_1 = Encoder(d_model,num_heads,dff_dim,rate)
enc_block_2 = Encoder(d_model,num_heads,dff_dim,rate)

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
#conv13_rdi = maxpool_1(conv13_rdi)

### Residual Block - 2
conv21_rdi = conv21_rdi(conv13_rdi)
conv22_rdi = conv22_rdi(conv21_rdi)
conv23_rdi = conv23_rdi(conv22_rdi)
conv23_rdi = tf.keras.layers.Add()([conv23_rdi,conv21_rdi])

##### ViViT
tubelet_embedding = tubelet_embedding_layer(conv23_rdi)
tokens = positional_embedding_encoder(tubelet_embedding)
enc_block_1_op = enc_block_1(tokens)
enc_block_2_op = enc_block_2(enc_block_1_op)

##### Output Layer
gap_op = tf.keras.layers.GlobalAveragePooling1D()(enc_block_2_op)
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
filepath= "./Models/GBQA_msRes3D-ViViT_CU-5050_Soli.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,monitor='val_accuracy',save_best_only=True,mode='max')

###### Training the Model
history = model.fit(X_train,y_train,epochs=100,batch_size=24,
                    validation_data=(X_dev,y_dev),validation_batch_size=24,
                   callbacks=checkpoint)

##### Saving Training Metrics
np.save('./Model History/GBQA_msRes3D-ViViT_CU-5050_Soli.npy',history.history)
 