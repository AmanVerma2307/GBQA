####### Importing Libraries
import itertools
import argparse
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.MViT_Encoder import MViT_Encoder
from utils.ViViT import Tubelet_Embedding, PositionEmbedding, Encoder
from sklearn.metrics import confusion_matrix

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
p_t = 2
p_h = 4
p_w = 4
n_t = (((T - p_t)//p_t)+1)
n_h = (((H - p_h)//p_h)+1)
n_w = (((W - p_w)//p_w)+1)
max_seq_len = int(n_t*(n_h)*(n_w))
pe_input = n_t*n_h*n_w
expansion_ratio = 4
rate = 0.3

##### Convolutional Layers

#### Res3DNet
conv11_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
conv12_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
conv13_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
maxpool_1 = tf.keras.layers.MaxPooling3D(pool_size=(1,2,2))

conv21_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')
conv22_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')
conv23_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')

##### ViViT

#### tokenization
tubelet_embedding_layer = Tubelet_Embedding(d_model,(p_t,p_h,p_w))
positional_embedding_encoder = PositionEmbedding(max_seq_len,d_model)

#### Stage-1
block_11 = MViT_Encoder(d_model,d_model*2,num_heads,(2,2,2),
                            (2,2,2),(3,3,3),(3,3,3),
                            rate=0.3,dff_dim=128)
block_12 = Encoder(d_model*2,num_heads,dff_dim,rate)

#### Stage-2
block_21 = MViT_Encoder(d_model*2,d_model*4,num_heads,(2,1,1),
                            (2,1,1),(1,1,1),(1,1,1),
                            rate=0.3,dff_dim=128*2)
block_22 = Encoder(d_model*4,num_heads,dff_dim,rate)

#### Stage-3
block_31 = MViT_Encoder(d_model*2,d_model*4,num_heads,(1,2,2),
                            (1,2,2),(1,1,1),(1,1,1),
                            rate=0.3,dff_dim=128)
block_32 = Encoder(d_model*4,num_heads,dff_dim,rate)

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
#### Embedding layers
tubelet_embedding = tubelet_embedding_layer(conv23_rdi)
tokens = positional_embedding_encoder(tubelet_embedding)

### Stage-1
block_11_op, block_11_shape = block_11(tokens,[n_t,n_h,n_w])
block_12_op = block_12(block_11_op)

### Stage-2
block_21_op, block_21_shape = block_21(block_12_op,block_11_shape)
block_22_op = block_22(block_21_op)

##### Output Layer
gap_op = tf.keras.layers.GlobalAveragePooling1D()(block_22_op)
dense1 = tf.keras.layers.Dense(256,activation='relu')(gap_op)
dropout1 = tf.keras.layers.Dropout(rate=0.2)(dense1)

### Softmax Output Layer
dense2 = tf.keras.layers.Dense(256,activation='relu')(dropout1)
dense3 = tf.keras.layers.Dense(6,activation='softmax')(dense2)

###### Compiling Model
model = tf.keras.models.Model(inputs=Input_Layer,outputs=dense3)
model.load_weights('./Models/GBQA_Res3D-MViT_CU-5050_Soli.h5') 
model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()

####### Model Testing

###### Testing Model - Accuracy Score
print(model.evaluate(X_dev,y_dev))
g_hgr = model.predict(X_dev,batch_size=4)
y_preds = np.argmax(g_hgr,axis=-1)

###### Confusion Matrix
cm = confusion_matrix(y_dev,y_preds)
total_gest = np.sum(cm,axis=-1)
cm_norm = cm/total_gest
print(cm_norm)

##### Saving Results
result_file = open('./Result Files/GBQA_Res3D-MViT_CU-5050_Soli.txt','w')
result_file.write(str(cm_norm))
result_file.close()

##### Plotting Heatmap

#### Heatmap Plotting Function
plt.rcParams["figure.figsize"] = [8,12]
def plot_heatmap(cm,filepath,classes,normalize=False,title='Avg. HGR Probabilities',cmap=plt.cm.Blues):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",fontsize='large',
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_GramMatrix(cm,filepath,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

#### Heatmap Plotting
filepath='./Graphs/Softmax Heatmap/GBQA_Res3D-MViT_CU-5050_Soli.png'
cm_plot_labels = ['Pinch index','Palm tilt','Fast Swipe','Push','Finger rub','Circle']
plot_heatmap(cm=np.around(cm_norm,2),filepath=filepath,classes=cm_plot_labels,normalize=False)

###### Testing Model - SoftMax Style               
def normalisation_layer(x):   
    return(tf.math.l2_normalize(x, axis=1, epsilon=1e-12))

predictive_model = tf.keras.models.Model(inputs=model.input,outputs=model.layers[-2].output)
predictive_model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#y_in = tf.keras.layers.Input((11,))

Input_Layer_rdi = tf.keras.layers.Input((T,H,W,C_rdi))
#Input_Layer_rai = tf.keras.layers.Input((T,H,W,C_rai))
op_1 = predictive_model(Input_Layer_rdi)
final_norm_op = tf.keras.layers.Lambda(normalisation_layer)(op_1)

testing_model = tf.keras.models.Model(inputs=Input_Layer_rdi,outputs=final_norm_op)
testing_model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

##### Nearest Neighbor Classification
from sklearn.neighbors import KNeighborsClassifier
Test_Embeddings = testing_model.predict(X_dev)

col_mean = np.nanmean(Test_Embeddings, axis=0)
inds = np.where(np.isnan(Test_Embeddings))
#print(inds)
Test_Embeddings[inds] = np.take(col_mean, inds[1])
np.savez_compressed('./Embeddings/GBQA_Res3D-MViT_CU-5050_Soli.npz',Test_Embeddings)

#### t-SNE Plots
### t-SNE Embeddings
tsne_X_dev = TSNE(n_components=2,perplexity=30,learning_rate=10,n_iter=10000,n_iter_without_progress=50).fit_transform(Test_Embeddings) # t-SNE Plots 

### Plotting
plt.rcParams["figure.figsize"] = [12,8]
for idx,color_index in zip(list(np.arange(6)),["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf","yellow"]):
    plt.scatter(tsne_X_dev[y_dev == idx, 0],tsne_X_dev[y_dev == idx, 1],s=55,color=color_index,edgecolors='k',marker='h')
plt.legend(['Pinch index','Palm tilt','Fast Swipe','Push','Finger rub','Circle'],loc='best',prop={'size': 12})
#plt.grid(b='True',which='both')
plt.savefig('./Graphs/tSNE/GBQA_Res3D-MViT_CU-5050_Soli.png')
