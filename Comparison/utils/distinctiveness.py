import numpy as np

def distinctiveness(embeddings, y, y_id, num_gestures, num_id):

    """
    Function to return distinct measures per gesture   

    INPUTS:-
    1) embeddings: Input embeddings of dimensions (N,d)
    2) y: Corresponding gesture label list
    3) y_id: Corresponding id list
    4) num_gestures: Total gestures in the dataset
    5) num_id: Total subjects in the dataset

    OUTPUTS:-
    1) dist_val: Distinctiveness values of the gestures of shape (num_gestures,)
    """

    dist_val = []

    for g_id in range(num_gestures):

        dist_val_curr = 0 # Variable to track distinctiveness within the gesture
        g_embeddings = [] # List to store embeddings of the current gesture label
        id_list = [] # List to store ID labels        
        for idx, idx_label in enumerate(y): # Gesture embedding collection
            if(idx_label == g_id):
                g_embeddings.append(embeddings[idx])
                id_list.append(y_id[idx])

        #### Global statistics
        mu_global = np.mean(g_embeddings,axis=0)
        var_global = np.var(g_embeddings,axis=0)

        #### Local statistics
        for subject_id in range(num_id):

            sub_embeddings = [] # Current subject's and gesture's collection
            for idx, idx_label in enumerate(id_list):
                if(idx_label == subject_id):
                    sub_embeddings.append(g_embeddings[idx])

            mu_local_curr = np.mean(sub_embeddings,axis=0)
            var_local_curr = np.var(sub_embeddings,axis=0)

            dist_val_curr = dist_val_curr + np.sum(np.abs(mu_global-mu_local_curr)/(np.sqrt(var_global+var_local_curr+1)))
        dist_val_curr = dist_val_curr/(num_id) # Average computation
        dist_val.append(dist_val_curr)

    return dist_val

if __name__ ==  "__main__":
    x = np.load('./GBQA_tdsNet_CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
    y = np.load('./y_dev_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
    y_id = np.load('./y_dev_id_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']-16
    dis_val = distinctiveness(x,y,y_id,6,10)
    print(dis_val)