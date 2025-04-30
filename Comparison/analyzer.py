import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.capacityEstimation import dgbqa

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    help="Dataset")
parser.add_argument('--prune',
                    type=bool,
                    default=False,
                    help="With/Without")
parser.add_argument('--mode',
                    type=str,
                    help="dgbqa/tsne/bars")
parser.add_argument('--bar_mode',
                    type=str,
                    help="bias/prune")
args = parser.parse_args()

if(args.dataset == 'scut'):

    if(args.prune == True):
        x_train = np.load('./embeddings/GBQA_tdsNet_CU-Prune-Train_SCUT.npz',allow_pickle=True)['arr_0']
        x_dev = np.load('./embeddings/GBQA_tdsNet_CU-Prune_SCUT.npz',allow_pickle=True)['arr_0']
        y_train = (np.load('./embeddings/y_train_GBQA-CU-5050-Prune_SCUT.npz',allow_pickle=True)['arr_0'])[:4272]
        y_dev = np.load('./embeddings/y_dev_GBQA-CU-5050-Prune_SCUT.npz',allow_pickle=True)['arr_0']
        y_train_id = (np.load('./embeddings/y_train_id_GBQA-CU-5050-Prune_SCUT.npz',allow_pickle=True)['arr_0'])[:4272]
        y_dev_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050-Prune_SCUT.npz',allow_pickle=True)['arr_0']  
        num_gestures = 3
        num_subjects = 50
        gesture_list = ['Rotate to Fist','Catch and Release','Bend Four Fingers']
        color_list = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        eer = [7.80,5.19,5.13]

    else:
        x_train = np.load('./embeddings/GBQA_tdsNet_CU-Train_SCUT.npz',allow_pickle=True)['arr_0']
        x_dev = np.load('./embeddings/GBQA_tdsNet_CU-5050_SCUT.npz',allow_pickle=True)['arr_0']
        y_train = (np.load('./embeddings/y_train_GBQA-CU-5050_SCUT.npz',allow_pickle=True)['arr_0'])[:8568]
        y_dev = np.load('./embeddings/y_dev_GBQA-CU-5050_SCUT.npz',allow_pickle=True)['arr_0'] 
        y_train_id = (np.load('./embeddings/y_train_id_GBQA-CU-5050_SCUT.npz',allow_pickle=True)['arr_0'])[:8568]
        y_dev_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050_SCUT.npz',allow_pickle=True)['arr_0'] 
        num_gestures = 6 
        num_subjects = 50
        gesture_list = ['Fist','Rotate to Fist','Catch and Release','Four Fingers','Bend Four Fingers','Fist Opening']
        color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        eer = [7.48, 7.60, 5.19, 5.21, 5.13, 5.17]
        gesture_selection = [1,2,4]

if(args.dataset == 'soli'):

    if(args.prune == True):
        x_train = np.load('./embeddings/GBQA_tdsNet_CU-5050-Train_Soli.npz',allow_pickle=True)['arr_0']
        x_dev = np.load('./embeddings/GBQA_tdsNet_CU-5050_Soli.npz',allow_pickle=True)['arr_0']
        y_train = np.load('./embeddings/y_train_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']
        y_dev = np.load('./embeddings/y_dev_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']
        y_train_id = np.load('./embeddings/y_train_id_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']
        y_dev_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']-5
        num_gestures = 6
        num_subjects = 5
        gesture_list = ['Pinch index','Palm tilt','Fast swipe','Push','Finger rub','Circle']
        color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        eer = [15.60,14.33,4.74,7.13,8.15,5.94]
    
    else:
        x_train = np.load('./embeddings/GBQA_tdsNet_CU-5050-Full-Train_Soli.npz',allow_pickle=True)['arr_0']
        x_dev = np.load('./embeddings/GBQA_tdsNet_CU-5050-Full_Soli.npz',allow_pickle=True)['arr_0']
        y_train = np.load('./embeddings/y_train_GBQA-CU-5050-Full_Soli.npz',allow_pickle=True)['arr_0']
        y_dev = np.load('./embeddings/y_dev_GBQA-CU-5050-Full_Soli.npz',allow_pickle=True)['arr_0']
        y_train_id = np.load('./embeddings/y_train_id_GBQA-CU-5050-Full_Soli.npz',allow_pickle=True)['arr_0']
        y_dev_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050-Full_Soli.npz',allow_pickle=True)['arr_0']-5    
        num_gestures = 11
        num_subjects = 5
        color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf","yellow"]
        gesture_list = ['Pinch index','Palm tilt','Finger Slider','Pinch pinky','Slow Swipe','Fast Swipe','Push','Pull','Finger rub','Circle','Palm hold']
        eer = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]
        gesture_selection = [0,1,5,6,8,9]

if(args.dataset == 'tiny'):

    if(args.prune == True):
        x_train = np.load('./embeddings/GBQA_tdsNet_CU-5050-Prune-Train_Tiny.npz',allow_pickle=True)['arr_0']
        x_dev = np.load('./embeddings/GBQA_tdsNet_CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
        y_train = np.load('./embeddings/y_train_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
        y_dev = np.load('./embeddings/y_dev_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
        y_train_id = np.load('./embeddings/y_train_id_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
        y_dev_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']-16
        num_gestures = 6
        num_subjects = 10
        color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        gesture_list = ["PinchIndex", "PalmTilt", "FastSwipeRL", "Push", "FingerRub", "Circle"]
        eer = [18.64,24.21,12.95,17.08,21.31,10.33]
        
    else:
        x_train = np.load('./embeddings/GBQA_tdsNet_CU-5050-Train_Tiny.npz',allow_pickle=True)['arr_0']
        x_dev = np.load('./embeddings/GBQA_tdsNet_CU-5050_Tiny.npz',allow_pickle=True)['arr_0']
        y_train = np.load('./embeddings/y_train_GBQA-CU-5050_Tiny.npz',allow_pickle=True)['arr_0']
        y_dev = np.load('./embeddings/y_dev_GBQA-CU-5050_Tiny.npz',allow_pickle=True)['arr_0']
        y_train_id = np.load('./embeddings/y_train_id_GBQA-CU-5050_Tiny.npz',allow_pickle=True)['arr_0']
        y_dev_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050_Tiny.npz',allow_pickle=True)['arr_0']-16
        num_gestures = 11
        num_subjects = 10
        color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf","yellow"]
        gesture_list = ['Pinch index','Palm tilt','Finger Slider','Pinch pinky','Slow Swipe','Fast Swipe','Push','Pull','Finger rub','Circle','Palm hold']
        eer = [18.64,24.21,26.71,15.78,13.51,12.95,17.08,19.23,21.31,10.33,37.54]
        gesture_selection = [0,1,5,6,8,9]

if(args.mode == 'tsne'):

    x_train = TSNE(n_components=2,perplexity=30,learning_rate=10,n_iter=10000,n_iter_without_progress=50).fit_transform(x_train)
    x_dev = TSNE(n_components=2,perplexity=30,learning_rate=10,n_iter=10000,n_iter_without_progress=50).fit_transform(x_dev)

    plt.rcParams["figure.figsize"] = [12,8]
    fig, axes = plt.subplots(nrows=1, ncols=1)

    for idx,color_index in zip(list(np.arange(num_gestures)),color_list):
        axes.scatter(x_train[y_train == idx, 0],x_train[y_train == idx, 1],s=55,color=color_index,edgecolors='k',marker='o')

    for idx,color_index in zip(list(np.arange(num_gestures)),color_list):
        axes.scatter(x_dev[y_dev == idx, 0],x_dev[y_dev == idx, 1],s=55,color=color_index,edgecolors='k',marker='X')

    legend_gesture = plt.legend(gesture_list,loc='best',prop={'size': 12})

    #marker = ['$\\bullet$','$\\times$']
    #marker_labels = ['Training','Testing']
    #legend_marker = plt.legend(marker_labels,
    #                           bbox_to_anchor=(0.35,1.02,1,0.2), 
    #                           loc="lower left",
    #                           ncol=2,
    #                           prop={'size': 12})
    #legend_marker.legend_handles[0].set_facecolor('black')
    #legend_marker.legend_handles[1].set_facecolor('black')
    
    axes.add_artist(legend_gesture)
    #axes.add_artist(legend_marker)

    plt.show()

if(args.mode == 'dgbqa'):
    for g_id, gesture_curr in enumerate(gesture_list):
        print('==============================================')
        dgbqa_score_curr, d_c_star_curr, d_cs_curr, dgbqa_score_wo_curr = dgbqa(x_dev,g_id,num_subjects,y_dev,y_dev_id)
        print('d_UNQ for '+str(gesture_curr)+' = '+str(d_c_star_curr))  
        print('d_VRB for '+str(gesture_curr)+' = '+str(d_cs_curr)) 

if(args.mode == 'bars'):

    if(args.bar_mode == 'bias'):

        if(args.dataset == 'soli'):
            x_train = np.load('./embeddings/GBQA_tdsNet_CU-5050-Train_Soli.npz',allow_pickle=True)['arr_0']
            x_dev = np.load('./embeddings/GBQA_tdsNet_CU-5050_Soli.npz',allow_pickle=True)['arr_0']
            y_train = np.load('./embeddings/y_train_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']
            y_dev = np.load('./embeddings/y_dev_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']
            y_train_id = np.load('./embeddings/y_train_id_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']
            y_dev_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']-5
            num_gestures = 6
            num_subjects = 5
            gesture_list = ['Pinch index','Palm tilt','Fast swipe','Push','Finger rub','Circle']
            color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            eer = [15.60,14.33,4.74,7.13,8.15,5.94]
            gbqa = [1.00,6.00,19.00,8.00,10.00,14.00]

            d_unq = []
            d_vrb = []

            for g_id, gesture_curr in enumerate(gesture_list):
                _, d_unq_curr, d_vrb_curr, _ = dgbqa(x_dev,g_id,num_subjects,y_dev,y_dev_id)
                d_unq.append(d_unq_curr)
                d_vrb.append(d_vrb_curr)

            d_unq = np.array(d_unq)
            d_vrb = np.array(d_vrb)

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            x_axes = np.arange(start=0,stop=12,step=2)
            ax.bar(x_axes,d_unq,zs=-0.05,zdir='y',color='dodgerblue',label='$d_{UNQ}$')
            ax.bar(x_axes+0.8,d_vrb,zs=-0.05,zdir='y',color='khaki',label='$d_{VRB}$')
            ax.bar(x_axes,eer,zs=0,color='pink',label='EER (%)')
            ax.bar(x_axes+0.8,gbqa,zs=0,color='lightblue', label='GBQA')
            ax.invert_yaxis()

            #ax.set_zlabel('$$',fontsize=12)
            ax.set_ylabel('Biometric performance',fontsize=12)
            ax.set_xlabel('$Gesture$',fontsize=12)

            ax.set_xticks(x_axes,labels=gesture_list,fontsize=5.5,rotation=15)
            ax.legend(frameon=True,fontsize=8)
            plt.show()

        if(args.dataset == 'tiny'):
            
            x_train = np.load('./embeddings/GBQA_tdsNet_CU-5050-Prune-Train_Tiny.npz',allow_pickle=True)['arr_0']
            x_dev = np.load('./embeddings/GBQA_tdsNet_CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
            y_train = np.load('./embeddings/y_train_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
            y_dev = np.load('./embeddings/y_dev_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
            y_train_id = np.load('./embeddings/y_train_id_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
            y_dev_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']-16
            num_gestures = 6
            num_subjects = 10
            color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            gesture_list = ["PinchIndex", "PalmTilt", "FastSwipeRL", "Push", "FingerRub", "Circle"]
            eer = [18.64,24.21,12.95,17.08,21.31,10.33]
            gbqa =  [16.00,2.00,9.00,1.00,10.00,10.00]

            d_unq = []
            d_vrb = []

            for g_id, gesture_curr in enumerate(gesture_list):
                _, d_unq_curr, d_vrb_curr, _ = dgbqa(x_dev,g_id,num_subjects,y_dev,y_dev_id)
                d_unq.append(d_unq_curr)
                d_vrb.append(d_vrb_curr)

            d_unq = np.array(d_unq)
            d_vrb = np.array(d_vrb)

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            x_axes = np.arange(start=0,stop=12,step=2)
            ax.bar(x_axes,d_unq,zs=-0.05,zdir='y',color='dodgerblue',label='$d_{UNQ}$')
            ax.bar(x_axes+0.8,d_vrb,zs=-0.05,zdir='y',color='khaki',label='$d_{VRB}$')
            ax.bar(x_axes,eer,zs=0,color='pink',label='EER (%)')
            ax.bar(x_axes+0.8,gbqa,zs=0,color='lightblue', label='GBQA')
            ax.invert_yaxis()

            #ax.set_zlabel('$$',fontsize=12)
            ax.set_ylabel('Biometric performance',fontsize=12)
            ax.set_xlabel('$Gesture$',fontsize=12)

            ax.set_xticks(x_axes,labels=gesture_list,fontsize=5.5,rotation=15)
            ax.legend(frameon=True,fontsize=8)
            plt.show()

        if(args.dataset == 'scut'):

            x_train = np.load('./embeddings/GBQA_tdsNet_CU-Prune-Train_SCUT.npz',allow_pickle=True)['arr_0']
            x_dev = np.load('./embeddings/GBQA_tdsNet_CU-Prune_SCUT.npz',allow_pickle=True)['arr_0']
            y_train = (np.load('./embeddings/y_train_GBQA-CU-5050-Prune_SCUT.npz',allow_pickle=True)['arr_0'])[:4272]
            y_dev = np.load('./embeddings/y_dev_GBQA-CU-5050-Prune_SCUT.npz',allow_pickle=True)['arr_0']
            y_train_id = (np.load('./embeddings/y_train_id_GBQA-CU-5050-Prune_SCUT.npz',allow_pickle=True)['arr_0'])[:4272]
            y_dev_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050-Prune_SCUT.npz',allow_pickle=True)['arr_0']  
            num_gestures = 3
            num_subjects = 50
            gesture_list = ['Rotate to Fist','Catch and Release','Bend Four Fingers']
            color_list = ["#1f77b4", "#ff7f0e", "#2ca02c"]
            eer = [7.80,5.19,5.13]
            gbqa =  [0.00,0.20,0.00]

            d_unq = []
            d_vrb = []

            for g_id, gesture_curr in enumerate(gesture_list):
                _, d_unq_curr, d_vrb_curr, _ = dgbqa(x_dev,g_id,num_subjects,y_dev,y_dev_id)
                d_unq.append(d_unq_curr)
                d_vrb.append(d_vrb_curr)

            d_unq = np.array(d_unq)
            d_vrb = np.array(d_vrb)

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            x_axes = np.arange(start=0,stop=6,step=2)
            ax.bar(x_axes,d_unq,zs=-0.05,zdir='y',color='dodgerblue',label='$d_{UNQ}$')
            ax.bar(x_axes+0.8,d_vrb,zs=-0.05,zdir='y',color='khaki',label='$d_{VRB}$')
            ax.bar(x_axes,eer,zs=0,color='pink',label='EER (%)')
            ax.bar(x_axes+0.8,gbqa,zs=0,color='lightblue', label='GBQA')
            ax.invert_yaxis()

            #ax.set_zlabel('$$',fontsize=12)
            ax.set_ylabel('Biometric performance',fontsize=12)
            ax.set_xlabel('$Gesture$',fontsize=8)

            ax.set_xticks(x_axes,labels=gesture_list,fontsize=5,rotation=15)
            ax.legend(frameon=True,fontsize=8)
            plt.show()

    if(args.bar_mode == 'prune'):

        if(args.dataset == 'soli'):

            x_train = np.load('./embeddings/GBQA_tdsNet_CU-5050-Train_Soli.npz',allow_pickle=True)['arr_0']
            x_dev = np.load('./embeddings/GBQA_tdsNet_CU-5050_Soli.npz',allow_pickle=True)['arr_0']
            y_train = np.load('./embeddings/y_train_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']
            y_dev = np.load('./embeddings/y_dev_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']
            y_train_id = np.load('./embeddings/y_train_id_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']
            y_dev_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']-5
            num_gestures = 6
            num_subjects = 5
            gesture_list = ['Pinch index','Palm tilt','Fast swipe','Push','Finger rub','Circle']

            d_unq = []
            d_vrb = []

            for g_id, gesture_curr in enumerate(gesture_list):
                _, d_unq_curr, d_vrb_curr, _ = dgbqa(x_dev,g_id,num_subjects,y_dev,y_dev_id)
                d_unq.append(d_unq_curr)
                d_vrb.append(d_vrb_curr)

            d_unq = np.array(d_unq)
            d_vrb = np.array(d_vrb)

            x_train = np.load('./embeddings/GBQA_tdsNet_CU-5050-Full-Train_Soli.npz',allow_pickle=True)['arr_0']
            x_dev = np.load('./embeddings/GBQA_tdsNet_CU-5050-Full_Soli.npz',allow_pickle=True)['arr_0']
            y_train = np.load('./embeddings/y_train_GBQA-CU-5050-Full_Soli.npz',allow_pickle=True)['arr_0']
            y_dev = np.load('./embeddings/y_dev_GBQA-CU-5050-Full_Soli.npz',allow_pickle=True)['arr_0']
            y_train_id = np.load('./embeddings/y_train_id_GBQA-CU-5050-Full_Soli.npz',allow_pickle=True)['arr_0']
            y_dev_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050-Full_Soli.npz',allow_pickle=True)['arr_0']-5    
            num_gestures = 11
            num_subjects = 5
            gesture_selection = [0,1,5,6,8,9]

            d_unq_wo = []
            d_vrb_wo = []

            for g_id in gesture_selection:
                _, d_unq_wo_curr, d_vrb_wo_curr, _ = dgbqa(x_dev,g_id,num_subjects,y_dev,y_dev_id)
                d_unq_wo.append(d_unq_wo_curr)
                d_vrb_wo.append(d_vrb_wo_curr)

            d_unq_wo = np.array(d_unq_wo)
            d_vrb_wo = np.array(d_vrb_wo)

            fig, (ax11, ax12) = plt.subplots(nrows=2, ncols=1, figsize=(4,6))
            x_axes = np.arange(start=0,stop=12,step=2)
            
            for ax in [ax11, ax12]:

                if(ax == ax11):
                    ax.bar(x_axes,d_unq_wo,color='pink',label='w/o Pruning')
                    ax.bar(x_axes+0.8,d_unq,color='lightblue',label='Pruning')
                    ax.set_ylabel('$d_{UNQ}$',fontsize=12)
                    #ax.set_xlabel('$Gesture$',fontsize=12)
                    ax.set_xticks(x_axes,labels=gesture_list,fontsize=12,rotation=15)
                    ax.legend(frameon=True,fontsize=8)

                if(ax == ax12):
                    ax.bar(x_axes,d_vrb_wo,color='pink',label='w/o Pruning')
                    ax.bar(x_axes+0.8,d_vrb,color='lightblue',label='Pruning')
                    ax.set_ylabel('$d_{VRB}$',fontsize=12)
                    ax.set_xlabel('$Gesture$',fontsize=12)
                    ax.set_xticks(x_axes,labels=gesture_list,fontsize=12,rotation=15)
                    ax.legend(frameon=True,fontsize=8)
                    
            plt.show()

        if(args.dataset == 'tiny'):

            x_train = np.load('./embeddings/GBQA_tdsNet_CU-5050-Prune-Train_Tiny.npz',allow_pickle=True)['arr_0']
            x_dev = np.load('./embeddings/GBQA_tdsNet_CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
            y_train = np.load('./embeddings/y_train_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
            y_dev = np.load('./embeddings/y_dev_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
            y_train_id = np.load('./embeddings/y_train_id_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
            y_dev_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']-16
            num_gestures = 6
            num_subjects = 10
            gesture_list = ["PinchIndex", "PalmTilt", "FastSwipeRL", "Push", "FingerRub", "Circle"]

            d_unq = []
            d_vrb = []

            for g_id, gesture_curr in enumerate(gesture_list):
                _, d_unq_curr, d_vrb_curr, _ = dgbqa(x_dev,g_id,num_subjects,y_dev,y_dev_id)
                d_unq.append(d_unq_curr)
                d_vrb.append(d_vrb_curr)

            d_unq = np.array(d_unq)
            d_vrb = np.array(d_vrb)

            x_train = np.load('./embeddings/GBQA_tdsNet_CU-5050-Train_Tiny.npz',allow_pickle=True)['arr_0']
            x_dev = np.load('./embeddings/GBQA_tdsNet_CU-5050_Tiny.npz',allow_pickle=True)['arr_0']
            y_train = np.load('./embeddings/y_train_GBQA-CU-5050_Tiny.npz',allow_pickle=True)['arr_0']
            y_dev = np.load('./embeddings/y_dev_GBQA-CU-5050_Tiny.npz',allow_pickle=True)['arr_0']
            y_train_id = np.load('./embeddings/y_train_id_GBQA-CU-5050_Tiny.npz',allow_pickle=True)['arr_0']
            y_dev_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050_Tiny.npz',allow_pickle=True)['arr_0']-16
            num_gestures = 11
            num_subjects = 10
            gesture_selection = [0,1,5,6,8,9]

            d_unq_wo = []
            d_vrb_wo = []

            for g_id in gesture_selection:
                _, d_unq_wo_curr, d_vrb_wo_curr, _ = dgbqa(x_dev,g_id,num_subjects,y_dev,y_dev_id)
                d_unq_wo.append(d_unq_wo_curr)
                d_vrb_wo.append(d_vrb_wo_curr)

            d_unq_wo = np.array(d_unq_wo)
            d_vrb_wo = np.array(d_vrb_wo)

            fig, (ax11, ax12) = plt.subplots(nrows=2, ncols=1, figsize=(4,6))
            x_axes = np.arange(start=0,stop=12,step=2)
            
            for ax in [ax11, ax12]:

                if(ax == ax11):
                    ax.bar(x_axes,d_unq_wo,color='pink',label='w/o Pruning')
                    ax.bar(x_axes+0.8,d_unq,color='lightblue',label='Pruning')
                    ax.set_ylabel('$d_{UNQ}$',fontsize=12)
                    #ax.set_xlabel('$Gesture$',fontsize=12)
                    ax.set_xticks(x_axes,labels=gesture_list,fontsize=12,rotation=15)
                    ax.legend(frameon=True,fontsize=8)

                if(ax == ax12):
                    ax.bar(x_axes,d_vrb_wo,color='pink',label='w/o Pruning')
                    ax.bar(x_axes+0.8,d_vrb,color='lightblue',label='Pruning')
                    ax.set_ylabel('$d_{VRB}$',fontsize=12)
                    ax.set_xlabel('$Gesture$',fontsize=12)
                    ax.set_xticks(x_axes,labels=gesture_list,fontsize=12,rotation=15)
                    ax.legend(frameon=True,fontsize=8)
                    
            plt.show()

        if(args.dataset == 'scut'):

            x_train = np.load('./embeddings/GBQA_tdsNet_CU-Prune-Train_SCUT.npz',allow_pickle=True)['arr_0']
            x_dev = np.load('./embeddings/GBQA_tdsNet_CU-Prune_SCUT.npz',allow_pickle=True)['arr_0']
            y_train = (np.load('./embeddings/y_train_GBQA-CU-5050-Prune_SCUT.npz',allow_pickle=True)['arr_0'])[:4272]
            y_dev = np.load('./embeddings/y_dev_GBQA-CU-5050-Prune_SCUT.npz',allow_pickle=True)['arr_0']
            y_train_id = (np.load('./embeddings/y_train_id_GBQA-CU-5050-Prune_SCUT.npz',allow_pickle=True)['arr_0'])[:4272]
            y_dev_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050-Prune_SCUT.npz',allow_pickle=True)['arr_0']  
            num_gestures = 3
            num_subjects = 50
            gesture_list = ['Rotate to Fist','Catch and Release','Bend Four Fingers']
            eer = [7.80,5.19,5.13]

            d_unq = []
            d_vrb = []

            for g_id, gesture_curr in enumerate(gesture_list):
                _, d_unq_curr, d_vrb_curr, _ = dgbqa(x_dev,g_id,num_subjects,y_dev,y_dev_id)
                d_unq.append(d_unq_curr)
                d_vrb.append(d_vrb_curr)

            d_unq = np.array(d_unq)
            d_vrb = np.array(d_vrb)

            x_train = np.load('./embeddings/GBQA_tdsNet_CU-Train_SCUT.npz',allow_pickle=True)['arr_0']
            x_dev = np.load('./embeddings/GBQA_tdsNet_CU-5050_SCUT.npz',allow_pickle=True)['arr_0']
            y_train = (np.load('./embeddings/y_train_GBQA-CU-5050_SCUT.npz',allow_pickle=True)['arr_0'])[:8568]
            y_dev = np.load('./embeddings/y_dev_GBQA-CU-5050_SCUT.npz',allow_pickle=True)['arr_0'] 
            y_train_id = (np.load('./embeddings/y_train_id_GBQA-CU-5050_SCUT.npz',allow_pickle=True)['arr_0'])[:8568]
            y_dev_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050_SCUT.npz',allow_pickle=True)['arr_0'] 
            num_gestures = 6 
            num_subjects = 50
            gesture_selection = [1,2,4]

            d_unq_wo = []
            d_vrb_wo = []

            for g_id in gesture_selection:
                _, d_unq_wo_curr, d_vrb_wo_curr, _ = dgbqa(x_dev,g_id,num_subjects,y_dev,y_dev_id)
                d_unq_wo.append(d_unq_wo_curr)
                d_vrb_wo.append(d_vrb_wo_curr)

            d_unq_wo = np.array(d_unq_wo)
            d_vrb_wo = np.array(d_vrb_wo)

            fig, (ax11, ax12) = plt.subplots(nrows=2, ncols=1, figsize=(4,6))
            x_axes = np.arange(start=0,stop=6,step=2)
            
            for ax in [ax11, ax12]:

                if(ax == ax11):
                    ax.bar(x_axes,d_unq_wo,color='pink',label='w/o Pruning')
                    ax.bar(x_axes+0.8,d_unq,color='lightblue',label='Pruning')
                    ax.set_ylabel('$d_{UNQ}$',fontsize=12)
                    #ax.set_xlabel('$Gesture$',fontsize=12)
                    ax.set_xticks(x_axes,labels=gesture_list,fontsize=12,rotation=15)
                    ax.legend(frameon=True,fontsize=8)

                if(ax == ax12):
                    ax.bar(x_axes,d_vrb_wo,color='pink',label='w/o Pruning')
                    ax.bar(x_axes+0.8,d_vrb,color='lightblue',label='Pruning')
                    ax.set_ylabel('$d_{VRB}$',fontsize=12)
                    ax.set_xlabel('$Gesture$',fontsize=12)
                    ax.set_xticks(x_axes,labels=gesture_list,fontsize=12,rotation=15)
                    ax.legend(frameon=True,fontsize=8)
                    
            plt.show()
            