import openface
import dlib
import sys
import caffe
import random
import math
from skimage import io
from skimage import draw
from skimage.draw import (line, polygon, circle,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)

import matplotlib.pyplot as plt 
import numpy as np
import argparse
import cv2
import os
import pickle
from headpose import findFace
from headpose import estimatePose
import multiprocessing
import sklearn.preprocessing as skpre
import csv

IMAGE_WIDTH = 140
IMAGE_HEIGHT = 40

def headPoseMatrixToEul(head_pose):
    norm = np.linalg.norm(head_pose)
    #print norm
    head_pose = head_pose / norm
    #print head_pose
    #print cv2.Rodrigues(head_pose)

    #print cv2.Rodrigues(cv2.Rodrigues(head_pose)[0])
    
    #print head_pose
    theta = np.float32(np.arcsin(head_pose[1][0]))
    phi = np.float32(np.arctan2(head_pose[0][0],head_pose[2][0]))
    #print theta, phi
    return theta, phi

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


if __name__ == '__main__':

    

    net = caffe.Net('deployEyeContact.prototxt',
                    caffe.TEST,
                    weights='snapshot__iter_100000.caffemodel',
                    )

    ### Change the file dir here!!!
    rootdir = "Eyes/Eyes_RA157"

    totalCount = 0
    relevant = 0
    irrlevant = 0
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0
    Tcount = 0
    Fcount = 0
    guess = 0


    predictions = []
    labels = []

    ### Change the ground truth file here!!!
    ifile  = open('RA157_groundtruth.csv', "rb")
    reader = csv.reader(ifile, delimiter=',')
    groundTruth = {}
    detected = []
    for row in reader:
        groundTruth[int(row[0])] = row[1]

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            #print file
            if not file[-3:] == "png":
                continue
            totalCount += 1
            img_path = rootdir + "\\" + file
            img = cv2.imread(img_path)
            img = transform_img(img)
            img = np.expand_dims(img, axis=2)
            img = np.rollaxis(img, 2)

            pkl_path = img_path[:-3] + "pickle"
            pkl_input = open(pkl_path, 'rb')
            labelDict = pickle.load(pkl_input)



            head_theta, head_phi = headPoseMatrixToEul(np.array(labelDict['rot']))
            #break
            headpose = (head_theta, head_phi)



            net.blobs['data'].data[...] = img
            net.blobs['headpose'].data[...] = headpose


            out = net.forward()
            ans = out["ip2"][0]
            contact = False

            
            idx = int(file[:-6])
            if groundTruth[idx] == 'U':
                continue

            detected.append(idx)
            if groundTruth[idx] == 'T':
                contact = True


            if ans > 0.5:
                if contact:
                    relevant += 1
                    truePositive += 1
                else:
                    irrlevant += 1
                    falsePositive += 1
            else:
                if contact:
                    relevant += 1
                    falseNegative += 1
                else:
                    irrlevant += 1
                    trueNegative += 1

    facesNum = 0
    
    for key in groundTruth.keys():
        key = int(key)
        
        if not key in detected:
            if groundTruth[key] == 'T':
                falseNegative += 1
                relevant += 1
            if groundTruth[key] == 'F':
                trueNegative += 1
                irrlevant += 1
        
        if groundTruth[key] is not 'U':
            facesNum += 1
    
    print relevant, irrlevant
    print truePositive, falsePositive, trueNegative, falseNegative

    Precision = float(truePositive) / float(truePositive + falsePositive)
    Recall = float(truePositive) / float(relevant)
    print "Detected Percentage: ", float(len(detected)) / float(facesNum)
    print "Precision: ", Precision 
    print "Recall: ", Recall 
    print "Accuracy: ", float(truePositive + trueNegative) / (relevant + irrlevant)
    print "F1: ", float(2 * truePositive) / float(2 * truePositive + falsePositive + falseNegative)