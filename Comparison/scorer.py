import argparse
import numpy as np
from utils.capacityEstimation import *
from utils.swipeQuality import *
from utils.distinctiveness import *
from utils.repeatability import *
from utils.rankDeviation import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    help="The dataset for which the results are to be computed")
parser.add_argument('--measure',
                    type=str,
                    default='all',
                    help='all/_name_of_measure')
args = parser.parse_args()

if(args.dataset == 'tiny'):
    num_gestures = 6
    num_ids = 10
    embeddings = np.load('./embeddings/GBQA_tdsNet_CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
    y = np.load('./embeddings/y_dev_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']
    y_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050-Prune_Tiny.npz',allow_pickle=True)['arr_0']-16
    gbqa = [16.00,2.00,9.00,1.00,10.00,10.00]
    eer = np.array([18.64,24.21,12.95,17.08,21.31,10.33])
    gesture_list = ['Pinch index','Palm tilt','Fast swipe','Push','Finger rub','Circle']

if(args.dataset == 'soli'):
    num_gestures = 6
    num_ids = 5
    embeddings = np.load('./embeddings/GBQA_tdsNet_CU-5050_Soli.npz',allow_pickle=True)['arr_0']
    y = np.load('./embeddings/y_dev_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']
    y_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050_Soli.npz',allow_pickle=True)['arr_0']-5
    gbqa = [1.00,6.00,19.00,8.00,10.00,14.00]
    eer = np.array([15.60,14.33,4.74,7.13,8.15,5.94])
    gesture_list = ['Pinch index','Palm tilt','Fast swipe','Push','Finger rub','Circle']

if(args.dataset == 'scut'):
    num_gestures = 6
    num_ids = 50
    embeddings = np.load('./embeddings/GBQA_tdsNet_CU-5050_SCUT.npz',allow_pickle=True)['arr_0']
    y = np.load('./embeddings/y_dev_GBQA-CU-5050_SCUT.npz',allow_pickle=True)['arr_0']
    y_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050_SCUT.npz',allow_pickle=True)['arr_0']
    gbqa = [1.5,0.7,0.6,0.2,0.2,0.6]
    eer = np.array([7.48, 7.60, 5.19, 5.21, 5.13, 5.17])
    gesture_list = ['Fist','Rotate to Fist','Catch and Release','Four Fingers','Bend Four Fingers','Fist Opening']

if(args.dataset == 'scut_prune'):
    num_gestures = 3
    num_ids = 50
    embeddings = np.load('./embeddings/GBQA_tdsNet_CU-Prune_SCUT.npz',allow_pickle=True)['arr_0']
    y = np.load('./embeddings/y_dev_GBQA-CU-5050-Prune_SCUT.npz',allow_pickle=True)['arr_0']
    y_id = np.load('./embeddings/y_dev_id_GBQA-CU-5050-Prune_SCUT.npz',allow_pickle=True)['arr_0']
    gbqa = [0.0,0.2,0.0]
    eer = np.array([7.60, 5.19, 5.13])
    gesture_list = ['Rotate to Fist','Catch and Release','Bend Four Fingers']

if(args.measure == 'all'):
    
    ###### Score estimation
    dgbqa_score = []
    d_c_star = []
    d_cs = []
    dgbqa_score_wo = []
    delta = []

    for g_id, gesture_curr in enumerate(gesture_list): # DGBQA
        dgbqa_score_curr, d_c_star_curr, d_cs_curr, dgbqa_score_wo_curr = dgbqa(embeddings,g_id,num_ids,y,y_id)
        dgbqa_score.append(dgbqa_score_curr)
        d_c_star.append(d_c_star_curr)
        d_cs.append(d_cs_curr)
        dgbqa_score_wo.append(dgbqa_score_wo_curr)

    for g_id, gesture_curr in enumerate(gesture_list): # Delta distance
        delta_curr = deltaDistance(embeddings,
                                            g_id,
                                            num_ids,
                                            y,
                                            y_id)
        delta.append(delta_curr)    

    generative = generativeCapacity(embeddings, 
                                            y,
                                            y_id,
                                            num_gestures,
                                            num_ids,
                                            32,
                                            0) # Generative capacity
    

    masterface = masterFace(np.array(d_c_star),32) # MasterFace
 
    swipe = swipeQuality(embeddings,y,num_gestures) # Swipe quality

    distinct = distinctiveness(embeddings,y,y_id,num_gestures,num_ids) # Distinctiveness

    repeat = repeatability(embeddings,y,y_id,num_gestures,num_ids) # Repeatability

    ##### Rank deviation
    rankDev_gbqa = rankDeviation(eer,np.array(gbqa),num_gestures)
    rankDev_dgbqa = rankDeviation(eer,np.array(dgbqa_score),num_gestures)
    rankDev_delta = rankDeviation(eer,np.array(delta),num_gestures)
    rankDev_gen = rankDeviation(eer,np.array(generative),num_gestures)
    rankDev_master = rankDeviation(eer,np.array(masterface),num_gestures)
    rankDev_swipe = rankDeviation(eer,-np.array(swipe),num_gestures)
    rankDev_distinct = rankDeviation(eer,np.array(distinct),num_gestures)
    rankDev_repeat = rankDeviation(eer,-np.array(repeat),num_gestures)

    ##### Results
    print('Swipe quality: '+str(rankDev_swipe))
    print('Distinctiveness: '+str(rankDev_distinct))
    print('Repeatability: '+str(rankDev_repeat))
    print('Delta: '+str(rankDev_delta))
    print('Generative capacity: '+str(rankDev_gen))
    print('MasterFace: '+str(rankDev_master))
    print('DGBQA: '+str(rankDev_dgbqa))
    print('GBQA: '+str(rankDev_gbqa))

else:    
    if(args.measure == 'gbqa'): # GBQA
        rankDev = rankDeviation(eer,np.array(gbqa),num_gestures)
    
    if(args.measure == 'dgbqa'): # DGBQA
        dgbqa_score = []
        d_c_star = []
        d_cs = []
        dgbqa_score_wo = []

        for g_id, gesture_curr in enumerate(gesture_list):
            dgbqa_score_curr, d_c_star_curr, d_cs_curr, dgbqa_score_wo_curr = dgbqa(embeddings,g_id,num_ids,y,y_id)
            dgbqa_score.append(dgbqa_score_curr)
            d_c_star.append(d_c_star_curr)
            d_cs.append(d_cs_curr)
            dgbqa_score_wo.append(dgbqa_score_wo_curr)

        rankDev = rankDeviation(eer,np.array(dgbqa_score),num_gestures)

    if(args.measure == 'delta'):
        delta = []
        for g_id, gesture_curr in enumerate(gesture_list):
            delta_curr = deltaDistance(embeddings,
                                                g_id,
                                                num_ids,
                                                y,
                                                y_id)
            delta.append(delta_curr)

        rankDev = rankDeviation(eer,np.array(delta),num_gestures)

    if(args.measure == 'generative'):
        generative = generativeCapacity(embeddings,
                                        y,
                                        y_id,
                                        num_gestures,
                                        num_ids,
                                        32,
                                        0)
        
        rankDev = rankDeviation(eer,np.array(generative),num_gestures)

    if(args.measure == 'masterface'):
        dgbqa_score = []
        d_c_star = []
        d_cs = []
        dgbqa_score_wo = []

        for g_id, gesture_curr in enumerate(gesture_list):
            dgbqa_score_curr, d_c_star_curr, d_cs_curr, dgbqa_score_wo_curr = dgbqa(embeddings,g_id,num_ids,y,y_id)
            dgbqa_score.append(dgbqa_score_curr)
            d_c_star.append(d_c_star_curr)
            d_cs.append(d_cs_curr)
            dgbqa_score_wo.append(dgbqa_score_wo_curr)

        masterface = masterFace(np.array(d_c_star),32)

        rankDev = rankDeviation(eer,masterface,num_gestures)

    if(args.measure == 'swipe'):
        swipe = swipeQuality(embeddings,y,num_gestures)
        rankDev = rankDeviation(eer,-np.array(swipe),num_gestures)

    if(args.measure == 'distinct'):
        distinct = distinctiveness(embeddings,y,y_id,num_gestures,num_ids)
        rankDev = rankDeviation(eer,np.array(distinct),num_gestures)
        
    if(args.measure == 'repeat'):
        repeat = repeatability(embeddings,y,y_id,num_gestures,num_ids)
        rankDev = rankDeviation(eer,-np.array(repeat),num_gestures)

    print('Rank deviation '+args.measure+' : '+str(rankDev))
