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

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

#return theta, phi
def headPoseMatrixToEul(head_pose):
    norm = np.linalg.norm(head_pose)
    #print norm
    head_pose = head_pose / norm
    print head_pose
    #print cv2.Rodrigues(head_pose)

    #print cv2.Rodrigues(cv2.Rodrigues(head_pose)[0])
    
    #print head_pose
    theta = np.float32(np.arcsin(head_pose[1][0]))
    phi = np.float32(np.arctan2(head_pose[0][0],head_pose[2][0]))
    #print theta, phi
    return theta, phi
    

rootdir = "Eyes"
hdf5prefix = "eyecontact_"
fileSize = 500
fileCounter = 0
idx = 0

print 'Creating hdf5'
f = h5py.File(hdf5prefix + str(fileCounter) + ".hdf5", "w")
dset = f.create_dataset("data", (fileSize, 1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype='f8', chunks=True)
lset = f.create_dataset("label", (fileSize, 3), dtype='f8', chunks=True)            
for subdir, dirs, files in os.walk(rootdir):

    for file in files:
        if file[-3:] != 'png':
            continue
        ###
        # Save Data
        ###    

        img_path = subdir + "\\" +  file
        img = cv2.imread(img_path)
        img = transform_img(img)
        img = np.expand_dims(img, axis=2)
        img = np.rollaxis(img, 2)

        # Read pkl
        pkl_path = img_path[:-3] + "pickle"
        pkl_input = open(pkl_path, 'rb')

        labelDict = pickle.load(pkl_input)

        # Convert Head Pose to Eul
        head_theta, head_phi = headPoseMatrixToEul(np.array(labelDict['rot']))
        #break
        head_pose = (head_theta, head_phi)
        
        # Get Eye Contact
        eyeContact = 0
        if file[-5] == 'T':
            eyeContact = 1
        elif file[-5] == 'F':
            eyeContact = 0
        else:
            print "error"
            print file

        label = [eyeContact, head_theta, head_phi]
        label = np.array(label)
        f['data'][idx] = img
        f['label'][idx] = label
        print file
        print label
        #break
        idx += 1

        if idx == fileSize:
            idx = 0
            fileCounter += 1
            print "Newfile" + str(fileCounter)
            f = h5py.File(hdf5prefix + str(fileCounter) + ".hdf5", "w")
            dset = f.create_dataset("data", (fileSize, 1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype='f8', chunks=True)
            lset = f.create_dataset("label", (fileSize, 3), dtype='f8', chunks=True)