import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.dataReader import gesture_seq_gen
from sklearn import mixture

test_folder = './Test001'
gest_seq_full = gesture_seq_gen(test_folder,None)

gest_seq = np.reshape(gest_seq_full,(-1,1))

gmm_model = mixture.GaussianMixture(n_components=2).fit(gest_seq)
gest_seq_mask = np.transpose(gmm_model.predict(gest_seq))
gest_seq_mask = np.reshape(gest_seq_mask,(60,250,250))

gest_seq_mask = 1 - gest_seq_mask

#print(gest_seq.shape)

#print(np.max(gest_seq),np.min(gest_seq))
#gest_seq_frame = gest_seq_mask[25,:,:]

for i in range(60):
    plt.imshow(gest_seq_mask[i,:,:]*gest_seq_full[i,:,:],cmap='gray')
    plt.show()

#print(gest_seq.shape)
#plt.imshow(gest_seq[5,:,:],cmap='gray')
#plt.show()

#fgbg = cv2.createBackgroundSubtractorMOG2()
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

#fgmask = fgbg.apply(gest_seq_frame)
#ret , treshold = cv2.threshold(fgmask.copy(), 120, 255,cv2.THRESH_BINARY)
#fgmask = cv2.dilate(fgmask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),iterations = 2)

#fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#print(fgmask.shape)

#plt.imshow(fgmask,cmap='gray')
#plt.show()