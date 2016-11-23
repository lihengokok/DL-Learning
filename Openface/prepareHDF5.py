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
IMAGE_WIDTH = 140
IMAGE_HEIGHT = 40


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    img = np.expand_dims(img, axis=2)
    img = np.rollaxis(img, 2)

    return img

def headPoseMatrixToEul(head_pose):

    #print head_pose
    head_pose = head_pose / np.pi
    #print head_pose

    #print head_pose
    theta = np.float32(np.arcsin(head_pose[1][0]))
    phi = np.float32(np.arctan2(head_pose[0][0],head_pose[2][0]))
    #print theta, phi
    return theta, phi



inputRootdir = "Eyes"
outputRootdir = "Eyes_HDF5"
hdf5TrainPrefix = "eye_contact_train"
fileSize = 1000

#Count File numbers
totalImagesCounter = 0
for subdir, dirs, files in os.walk(inputRootdir):
    print subdir
    for file in files:
        if file[-3:] == 'png':
            totalImagesCounter += 1

print "Total_Images: ", totalImagesCounter

#Create HDF5Files
hdf5Files = {}
fileCounter = {}

# Create Train files
for i in range(totalImagesCounter / fileSize + 1):
    fileName = outputRootdir + '\\' + hdf5TrainPrefix + str(i) + ".hdf5"
    fileCounter[i] = 0
    f = h5py.File(fileName, "w")
    dset = f.create_dataset("data", (fileSize, 1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype='f8', chunks=True)
    lset = f.create_dataset("label", (fileSize, 3), dtype='f8', chunks=True)   
    hdf5Files[i] = f


print "Total files: ", len(hdf5Files)

print "Filling hdf5 files..."

currentCounter = 0

for subdir, dirs, files in os.walk(inputRootdir):
    print "Processing Folder: ", subdir, "..."
    for file in files:
        if file[-3:] != 'png':
            continue
        img_path = subdir + "\\" +  file

        img = cv2.imread(img_path)
        img = transform_img(img)
        
        
        # Read pkl
        pkl_path = img_path[:-3] + "pickle"
        pkl_input = open(pkl_path, 'rb')
        labelDict = pickle.load(pkl_input)

        # Convert Head Pose to Eul
        head_theta, head_phi = headPoseMatrixToEul(np.array(labelDict['rot']))
        head_pose = (head_theta, head_phi)

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


        # Generate a random number for 0 to total number of files
        fileIndex = random.randint(0, len(hdf5Files) - 1)

        fillFromStart = False
        # If the files is full
        if fileCounter[fileIndex] == fileSize:
            for i in hdf5Files.keys():
                if fileCounter[i] < fileSize:
                    fileIndex = i
                    break

        idx = fileCounter[fileIndex] 
        hdf5Files[fileIndex]['data'][idx] = img
        hdf5Files[fileIndex]['label'][idx] = label
        fileCounter[fileIndex] += 1

        currentCounter += 1
        if currentCounter % 100 == 0:
            print "Processed Files: ", currentCounter
        


