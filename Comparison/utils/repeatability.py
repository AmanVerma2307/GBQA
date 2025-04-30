import numpy as np
from scipy.spatial import distance

def repeatability(embeddings, y, y_id, num_gestures, num_id):

    """
    Function to return distinct measures per gesture   

    INPUTS:-
    1) embeddings: Input embeddings of dimensions (N,d)
    2) y: Corresponding gesture label list
    3) y_id: Corresponding id list
    4) num_gestures: Total gestures in the dataset
    5) num_id: Total subjects in the dataset

    OUTPUTS:-
    1) repeat_val: Distinctiveness values of the gestures of shape (num_gestures,)
    """

    repeat_val = []

    for g_id in range(num_gestures):

        repeat_val_curr = [] # Variable to track distinctiveness within the gesture
        g_embeddings = [] # List to store embeddings of the current gesture label
        id_list = [] # List to store ID labels        
        for idx, idx_label in enumerate(y): # Gesture embedding collection
            if(idx_label == g_id):
                g_embeddings.append(embeddings[idx])
                id_list.append(y_id[idx])

        #### Local statistics
        for subject_id in range(num_id):

            repeat_val_curr_subject = 0

            sub_embeddings = [] # Current subject's and gesture's collection
            for idx, idx_label in enumerate(id_list):
                if(idx_label == subject_id):
                    sub_embeddings.append(g_embeddings[idx])
            sub_embeddings = np.array(sub_embeddings)

            mu_template = np.mean(np.array(sub_embeddings),axis=0)

            for item in sub_embeddings:
                repeat_val_curr_subject = repeat_val_curr_subject + distance.euclidean(item,mu_template)
            
            repeat_val_curr_subject = repeat_val_curr_subject/sub_embeddings.shape[0] # Average distance from the template

            repeat_val_curr.append(repeat_val_curr_subject)

        repeat_val.append(np.mean(repeat_val_curr))

    return repeat_val

if __name__ ==  "__main__":
    x = np.load('./GBQA_tdsNet_CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
    y = np.load('./y_dev_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
    y_id = np.load('./y_dev_id_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']-16
    rep_val = repeatability(x,y,y_id,6,10)
    print(rep_val)