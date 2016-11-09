import os
import glob
import random
import numpy as np
import cv2
import caffe
import pickle
from caffe.proto import caffe_pb2
import lmdb
import matplotlib.pyplot as plt
import h5py
IMAGE_WIDTH = 120
IMAGE_HEIGHT = 80
train_lmdb = 'syn_train_lmdb'
validation_lmdb = 'syn_validation_lmdb'

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

#return theta, phi
def headPoseMatrixToEul(head_pose):
    ZVector = cv2.Rodrigues(head_pose)[0]
    theta = np.float32(np.arcsin(ZVector[1])[0])
    phi = np.float32(np.arctan2(ZVector[0][0],ZVector[2][0]))
    return theta, phi

# return theta, phi
def gazeToEul(gaze):
    return np.arcsin(gaze[1] * -1), np.arctan2(gaze[0] * -1, gaze[2] * -1)




os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)

f = h5py.File("testfilet.hdf5", "w")
dset = f.create_dataset("data", (2000, 3, IMAGE_WIDTH, IMAGE_HEIGHT), dtype='i', chunks=True)
lset = f.create_dataset("label", (2000, 4), dtype='f8', chunks=True)
print dset.shape, dset.size


train_data = [img for img in glob.glob("f02/*.png")]

print 'Creating hdf5'


for idx, img_path in enumerate(train_data):
    img = cv2.imread(img_path)
    img = transform_img(img)
    img = np.rot90(img)
    img = np.rollaxis(img, 2)

    

    # Read pkl
    pkl_path = img_path[:-3] + "pkl"
    pkl_input = open(pkl_path, 'rb')
    labelDict = pickle.load(pkl_input)

    # Convert Head Pose to Eul
    head_theta, head_phi = headPoseMatrixToEul(np.array(labelDict['head_pose']))
    head_pose = (head_theta, head_phi)
    
    # Conver Gaze to Eul
    gaze_theta, gaze_phi = gazeToEul(np.array(labelDict['look_vec']))
    gaze = (gaze_theta, gaze_phi)

    # concate label
    label = [gaze_theta, gaze_phi, head_theta, head_phi]
    #print label
    # print label.shape
    # put data
    f['data'][idx] = img
    f['label'][idx] = label

        
        



