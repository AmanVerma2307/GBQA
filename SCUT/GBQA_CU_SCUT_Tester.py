####### Importing Libraries
import itertools
import argparse
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tdsNet import *
plt.switch_backend('agg')

####### Model Arguments and Hyperparameters
parser = argparse.ArgumentParser()

parser.add_argument('--fold_id',
                    type=str,
                    help="Testing subject id")
parser.add_argument('--exp_name',
                    type=str,
                    help="Experiment name")

args = parser.parse_args()

####### Loading Dataset
X_train = np.load('./Datasets/HandLogin/GBQA/X_train_GBQA-CU-'+args.fold_id+'SCUT.npz',allow_pickle=True)['arr_0']
X_dev = np.load('./Datasets/HandLogin/GBQA/X_dev_GBQA-CU-'+args.fold_id+'SCUT.npz',allow_pickle=True)['arr_0']
y_train = np.load('./Datasets/HandLogin/GBQA/y_train_GBQA-CU-'+args.fold_id+'SCUT.npz',allow_pickle=True)['arr_0']
y_dev = np.load('./Datasets/HandLogin/GBQA/y_dev_GBQA-CU-'+args.fold_id+'SCUT.npz',allow_pickle=True)['arr_0']
y_train_id = np.load('./Datasets/HandLogin/GBQA/y_train_id_GBQA-CU-'+args.fold_id+'SCUT.npz',allow_pickle=True)['arr_0']
y_dev_id = np.load('./Datasets/HandLogin/GBQA/y_dev_id_GBQA-CU-'+args.fold_id+'SCUT.npz',allow_pickle=True)['arr_0']

print(X_train.shape)
print(X_dev.shape)

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

####### Model Testing

###### Testing Model - Accuracy Score
g_hgr = model.predict(X_dev,batch_size=4)
y_preds = np.argmax(g_hgr,axis=-1)

#### Saving Predictions
print(model.evaluate(X_dev,y_dev,batch_size=4))
np.savez_compressed('./Predictions/'+args.exp_name+'.npz',y_preds)

###### Softmax Heatmap
###### Computing Avg. Probability Scores
y_preds_hgr_probs = np.zeros((6,6))

##### Iterating over the Predicted Probabilites
for g_idx in range(6):

    g_prob = [] # List to store the Predicted Probabilties of the Current Class

    for idx, y_val_idx in enumerate(y_dev): # Iterating over the Dataset
        
        if(y_val_idx == g_idx):
            g_prob.append(g_hgr[idx])

    g_prob = np.around(np.mean(np.array(g_prob),axis=0),decimals=2)
    y_preds_hgr_probs[g_idx,:] = g_prob

print(y_preds_hgr_probs)

##### Saving Results
result_file = open('./Result Files/'+args.exp_name+'.txt','w')
result_file.write(str(y_preds_hgr_probs))
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
cm_plot_labels = ['Fist','Rotate to Fist','Catch and Release','Four Fingers','Bend Four Fingers','Fist Opening']
plot_heatmap(cm=np.around(y_preds_hgr_probs,2),filepath=filepath,classes=cm_plot_labels,normalize=False)

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
for idx,color_index in zip(list(np.arange(6)),["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]):
    plt.scatter(tsne_X_dev[y_dev == idx, 0],tsne_X_dev[y_dev == idx, 1],s=55,color=color_index,edgecolors='k',marker='h')
plt.legend(['Fist','Rotate to Fist','Catch and Release','Four Fingers','Bend Four Fingers','Fist Opening'],loc='best',prop={'size': 12})
#plt.grid(b='True',which='both')
plt.savefig('./Graphs/tSNE/'+args.exp_name+'.png')
