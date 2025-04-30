####### Importing Libraries
import argparse
import numpy as np
import tensorflow as tf
from ViViT import Tubelet_Embedding, PositionEmbedding, Encoder

####### Model Arguments and Hyperparameters
parser = argparse.ArgumentParser()

parser.add_argument('--prune',
                    type=bool,
                    default=False,
                    help="Pruned data or not")
parser.add_argument('--exp_name',
                    type=str,
                    help="Experiment/model name")

args = parser.parse_args()

if(args.prune == True):
    X_train = np.load('./Datasets/TinyRadar/GBQA/X_train_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
    X_dev = np.load('./Datasets/TinyRadar/GBQA/X_dev_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
    y_train = np.load('./Datasets/TinyRadar/GBQA/y_train_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
    y_dev = np.load('./Datasets/TinyRadar/GBQA/y_dev_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']

    num_gestures = 6

    print(X_train.shape)
    print(X_dev.shape)
    print(y_train.shape)
    print(y_dev.shape)

if(args.prune == False):
    X_train = np.load('./Datasets/TinyRadar/GBQA/X_train_GBQA-CU-5050_Tiny.npz',allow_pickle=True)['arr_0']
    X_dev = np.load('./Datasets/TinyRadar/GBQA/X_dev_GBQA-CU-5050_Tiny.npz',allow_pickle=True)['arr_0']
    y_train = np.load('./Datasets/TinyRadar/GBQA/y_train_GBQA-CU-5050_Tiny.npz',allow_pickle=True)['arr_0']
    y_dev = np.load('./Datasets/TinyRadar/GBQA/y_dev_GBQA-CU-5050_Tiny.npz',allow_pickle=True)['arr_0']

    num_gestures = 11

    print(X_train.shape)
    print(X_dev.shape)
    print(y_train.shape)
    print(y_dev.shape)

###### Custom Model Checkpointing
class ModelCheckpointing_AccLoss(tf.keras.callbacks.Callback):

    """
     Callback to save the model with best validation accuracy and 
     mininmum validation loss.
     First preference is accuracy, and then for breaking ties 
     validation loss is used.
    """

    def __init__(self,filepath):
        
        ##### Defining Essentials    
        super(ModelCheckpointing_AccLoss, self).__init__()
        self.best_acc = 0.0  # Initializing with Zero Accuracy
        self.best_loss = np.inf # Initializing with Infinite Loss
        self.filepath = filepath # Path of the File wherein weights are to be saved

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):

        #### Logging Current Values
        acc_curr = logs['val_accuracy']
        loss_curr = logs['val_loss']

        #### Saving Weights
        if(acc_curr > self.best_acc):
            self.model.save_weights(self.filepath) # Saving Model
            self.best_acc = acc_curr # Updating best accuracy
            self.best_loss = loss_curr # Updating current loss
            #print('Saved the model with the highest validation accuracy')

        elif(acc_curr == self.best_acc): # If tie between accuracies

            if(loss_curr <  self.best_loss): # Using validation loss to break the ties
                self.model.save_weights(self.filepath) # Saving Model
                self.best_acc = acc_curr # Updating best accuracy
                self.best_loss = loss_curr # Updating current loss
                #print('Saved the model with the highest validation accuracy and lowest validation loss')

        else:
            return
        
####### Model Training
###### Defining Layers and Model

###### Defining Essentials
T = 5
H = 32
W = 492
C_rdi = 2
num_layers = 2
d_model = 32
num_heads = 16
dff_dim = 128
p_t = 2
p_h = 5
p_w = 15
n_t = (((T - p_t)//p_t)+1)
n_h = (((H - p_h)//p_h)+1)
n_w = (((W - p_w)//p_w)+1)
max_seq_len = n_t*n_h*n_w
pe_input = n_t*n_h*n_w
rate = 0.3

###### Defining Layers

##### Convolutional Layers

#### Res3DNet
conv11_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
conv12_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
conv13_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
maxpool_1 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2))

conv21_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')
conv22_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')
conv23_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')

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

#####  ViViT
tubelet_embedding = tubelet_embedding_layer(conv23_rdi)
tokens = positional_embedding_encoder(tubelet_embedding)
enc_block_1_op = enc_block_1(tokens)
enc_block_2_op = enc_block_2(enc_block_1_op)

##### Output Layer
gap_op = tf.keras.layers.GlobalAveragePooling1D()(enc_block_2_op)
dense1 = tf.keras.layers.Dense(32,activation='relu')(gap_op)
dense2 = tf.keras.layers.Dense(num_gestures,activation='softmax')(dense1)

###### Compiling Model
model = tf.keras.models.Model(inputs=Input_Layer,outputs=dense2) 
model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()
#tf.keras.utils.plot_model(model)

##### Defining Callbacks
filepath= './Models/'+args.exp_name+'.h5'
checkpoint = ModelCheckpointing_AccLoss(filepath)

###### Training the Model
history = model.fit(X_train,y_train,epochs=70,batch_size=32,
                validation_data=(X_dev,y_dev), validation_batch_size=32,
                   callbacks=checkpoint)

##### Saving Training Metrics
np.save('./Model History/'+args.exp_name+'.npy',history.history)
