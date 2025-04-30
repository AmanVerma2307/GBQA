import numpy as np

def swipeQuality(embeddings, y, num_gestures):

    """
    Function to return quality measures per gesture   

    INPUTS:-
    1) embeddings: Input embeddings of dimensions (N,d)
    2) y: Corresponding gesture label list
    3) num_gestures: Total gestures in the dataset

    OUTPUTS:-
    1) quality_val: Quality values of the gestures of shape (num_gestures,)
    """

    ##### Global statistics
    mu_global = np.mean(embeddings,axis=0)
    sigma_global = np.var(embeddings,axis=0)

    ##### Local statisitics
    quality_val = []
    d = embeddings.shape[-1]

    for g_id in range(num_gestures):

        curr_gest_embedds = [] # List to store embeddings of the current gesture

        for idx, emb_curr in enumerate(embeddings): # Collecting gesture of current 'g_id'
            if(y[idx] == g_id): 
                curr_gest_embedds.append(emb_curr)

        #### Statistic estimation
        curr_gest_embedds = np.array(curr_gest_embedds)
        mu_local = np.mean(curr_gest_embedds,axis=0)
        sigma_local = np.var(curr_gest_embedds,axis=0)

        #### Quality estimation
        quality_val_curr = np.sum(np.abs(mu_global-mu_local)/(np.sqrt(sigma_global+sigma_local+1)))
        quality_val.append(quality_val_curr)
            
    return quality_val

if __name__ ==  "__main__":
    x = np.load('./GBQA_tdsNet_CU-5050_SCUT.npz',allow_pickle=True)['arr_0']
    y = np.load('./y_dev_CU-5050_SCUT.npz',allow_pickle=True)['arr_0']
    quality_val = swipeQuality(x,y,6)
    print(quality_val)