import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

embeddings = np.load('./DGBQA_HGR_TDSNet_SOLI.npz',allow_pickle=True)['arr_0']
y_dev = np.load('./y_dev_DGBQA-Seen_SOLI.npz',allow_pickle=True)['arr_0']
y_dev_id = np.load('./y_dev_id_DGBQA-Seen_SOLI.npz',allow_pickle=True)['arr_0']

#### t-SNE Plots
### t-SNE Embeddings
tsne_X_dev = TSNE(n_components=2,perplexity=30,learning_rate=10,n_iter=10000,n_iter_without_progress=50).fit_transform(embeddings) # t-SNE Plots 

markers = ["o","1","v","8","s","p","P","*","X","d","H"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf","yellow"]

plt.rcParams["figure.figsize"] = [6,4]
for idx, emb_curr in enumerate(tsne_X_dev):
    if(int(y_dev[idx]) in [9]):    
        plt.scatter(emb_curr[0],emb_curr[1],s=75,color=colors[int(y_dev_id[idx])],edgecolors='k',marker=markers[int(y_dev_id[idx])])
plt.show()

### Plotting
#plt.rcParams["figure.figsize"] = [12,8]
#for idx,color_index in zip(list(np.arange(10)),["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf","yellow"]):
#    plt.scatter(tsne_X_dev[y_dev == idx, 0],tsne_X_dev[y_dev == idx, 1],s=55,color=color_index,edgecolors='k',marker="H")
#plt.legend(['Pinch index','Palm tilt','Finger Slider','Pinch pinky','Slow Swipe','Fast Swipe','Push','Pull','Finger rub','Circle','Palm hold'],loc='best',prop={'size': 12})
#plt.grid(b='True',which='both')
#plt.show()

#for s_idx, s_marker in zip(list(np.arange(10)),["o","1","v","8","s","p","P","*","X","d"]):