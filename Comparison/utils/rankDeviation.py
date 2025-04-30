import numpy as np

###### Rank Derivation
def rank_compute(rank_vector,val_to_rank,reverse_flag):

    """
    Function to derive rank of a particular value

    INPUTS:-
    1) rank_vector: Vector in which ranking is to be performed
    2) val_to_rank: Value whoose rank is to be derived
    3) reverse_flag: Flag to signify if the order of sort is to be reversed

    OUTPUTS:-
    1) rank_val: Ranked Value - The best is rank '0'
    """ 

    if(reverse_flag == True):
        rank_vector_sort = -(np.sort(-rank_vector))
    else:
        rank_vector_sort = np.sort(rank_vector) # Sorting the Vector    
    for item_idx,item in enumerate(rank_vector_sort): # Iterating over the vector
        if(item == val_to_rank): # Match-Found
            rank_val = item_idx # Rank Assignment 
            break

    return rank_val

###### Avg. Rank Deviation
def rankDeviation(eer_vec,dgbqa_vec,G_Total):
    
    """
    Computation of Acceptance Score
    
    INPUTS:-
    1) eer_vec: Vector of EER values
    2) dgbqa_vec: Vector of DGBQA values
    3) G_Total: Total gestures in the vector
    
    OUTPUTS:-
    1) avg_rank_deviation: Total Deviation in Rank/Total Number of Gestures
    """
    
    rank_deviation = 0 # Total Rank Deviation
    
    for g_idx in range(G_Total):
        
        #### Values of the Current Gesture
        eer_val_curr = eer_vec[g_idx] # EER Value
        dgbqa_val_curr = dgbqa_vec[g_idx] # DGBQA Value
        
        #### Rank of the Current Gesture
        eer_rank_curr = rank_compute(eer_vec,eer_val_curr,True) # EER Rank
        dgbqa_rank_curr = rank_compute(dgbqa_vec,dgbqa_val_curr,False) # DGBQA Rank
        
        #### Rank Difference
        rank_deviation =  rank_deviation + np.abs(eer_rank_curr - dgbqa_rank_curr)
        
    avg_rank_deviation = rank_deviation/G_Total # Computing Avg. over all the gestures
    return avg_rank_deviation