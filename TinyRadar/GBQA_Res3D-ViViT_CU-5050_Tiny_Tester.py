####### Importing Libraries
import itertools
import argparse
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
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
    gestures = ["PinchIndex", "PalmTilt", "FastSwipeRL", "Push", "FingerRub", "Circle"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

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
    gestures = ["PinchIndex", "PalmTilt", "FingerSlider", "PinchPinky", "SlowSwipeRL", "FastSwipeRL", "Push", "Pull", "FingerRub", "Circle", "PalmHold"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf","yellow"]

    print(X_train.shape)
    print(X_dev.shape)
    print(y_train.shape)
    print(y_dev.shape)

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
model.load_weights('./Models/'+args.exp_name+'.h5') 
model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

####### Model Testing

###### Testing Model - Accuracy Score
g_hgr = model.predict(X_dev,batch_size=4)
y_preds = np.argmax(g_hgr,axis=-1)

#### Saving Predictions
print(model.evaluate(X_dev,y_dev,batch_size=4))
np.savez_compressed('./Predictions/'+args.exp_name+'.npz',y_preds)

###### Confusion Matrix
cm = confusion_matrix(y_dev,y_preds)
total_gest = np.sum(cm,axis=-1)
cm_norm = cm/total_gest
print(cm_norm)

##### Saving Results
result_file = open('./Result Files/'+args.exp_name+'.txt','w')
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
filepath='./Graphs/Softmax Heatmap/'+args.exp_name+'.png'
cm_plot_labels = gestures
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

##Input_Layer_Flipped = tf.keras.layers.Input((224,224,3))
##op_2 = predictive_model([Input_Layer_Flipped,y_in]) 
##final_op = tf.keras.layers.Concatenate(axis=1)(op_1)

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

#### Computing and Saving Output 
np.savez_compressed('./Embeddings/'+args.exp_name+'.npz',Test_Embeddings)

#### t-SNE Plots
### t-SNE Embeddings
tsne_X_dev = TSNE(n_components=2,perplexity=30,learning_rate=10,n_iter=10000,n_iter_without_progress=50).fit_transform(Test_Embeddings) # t-SNE Plots 

### Plotting
plt.rcParams["figure.figsize"] = [12,8]
for idx,color_index in zip(list(np.arange(num_gestures)),colors):
    plt.scatter(tsne_X_dev[y_dev == idx, 0],tsne_X_dev[y_dev == idx, 1],s=55,color=color_index,edgecolors='k',marker='h')
plt.legend(gestures,loc='best',prop={'size': 12})
#plt.grid(b='True',which='both')
plt.savefig('./Graphs/tSNE/'+args.exp_name+'.png')