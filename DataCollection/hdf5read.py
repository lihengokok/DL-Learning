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
import Image

with h5py.File('syntheyes_0.hdf5', 'r') as hf:
    data = hf.get('data')
    label = hf.get('label')
    np_data = np.array(data)
    labeldata = np.array(label)
    print('Shape of the image: \n', np_data.shape)
    print('Shape of the label: \n', labeldata.shape)
    img = np_data[0]
    img = np.rot90(img)
    img = np.rollaxis(img, 2)
    newI = img * 255
    cv2.imwrite('example.png', newI)

    gaze = label[0][0:2]
    head = label[0][2:4]
    print 'Gaze: ', gaze, 'Head Pose:', head
    