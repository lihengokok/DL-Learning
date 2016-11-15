import openface
import dlib
import sys
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


def ang(lineA):
    vA = [(lineA[1][0]-lineA[0][0]), (lineA[1][1]-lineA[0][1]) * -1]
    return math.degrees(math.atan2(vA[1], vA[0])) * -1




def rotateImg(img, landmarks):
    LE = landmarks[36]
    RE = landmarks[45]
    center = ((LE[0] + RE[0]) / 2, (LE[1] + RE[1]) / 2)
    VA = [center, RE]
    angle = ang(VA)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    img = cv2.warpAffine(img, rot_mat, (1280, 720))
    return img


def cropEyeImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = dlib.get_frontal_face_detector()
    align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
    bbs = detector(img)
    if len(bbs) == 0:
        return None

    # Find the lowest face in the image (assuming it's the child's face)
    # Rotate image
    bb = None
    lowestY = 0
    for box in bbs:
        y1, y2 = box.top(), box.bottom()
        if (y1 + y2) / 2 > lowestY:
            lowestY = (y1 + y2) / 2
            bb = box

    originFace = bb

    landmarks = align.findLandmarks(img, bb)
    img = rotateImg(img, landmarks)
    if bb == None:
        return None
    # Crop Eye
    bb = None
    lowestY = 0
    for box in bbs:
        y1, y2 = box.top(), box.bottom()
        if (y1 + y2) / 2 > lowestY:
            lowestY = (y1 + y2) / 2
            bb = box
    landmarks = align.findLandmarks(img, bb)
    lel = landmarks[36]
    rer = landmarks[45]

    # Left eye
    width = math.ceil((landmarks[39][0] - landmarks[36][0]) * 1.65)
    height = width * 2 / 3
  
    LA = ((landmarks[17][0] + landmarks[36][0]) / 2, landmarks[17][1] + 0.25 * (landmarks[36][1] - landmarks[17][1]))
    LB = (LA[0] + width, LA[1] + height)
    x1 = int(LA[0])
    y1 = int(LA[1])
    x2 = int(LB[0])
    y2 = int(LB[1])
    lx1 = x1

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    leftEyeImg = img[y1:y2, x1:x2]
    leftEyeImg = cv2.resize(leftEyeImg, (60, 40))

    # Right eye
    width = math.ceil((landmarks[45][0] - landmarks[42][0]) * 1.65)
    height = width * 2 / 3

    LA = ((landmarks[26][0] + landmarks[45][0]) / 2, landmarks[26][1] + 0.25 * (landmarks[45][1] - landmarks[26][1]))
    LB = (LA[0] - width, LA[1] + height)
    x1 = int(LB[0])
    y1 = int(LA[1])
    x2 = int(LA[0])
    y2 = int(LB[1])



    rightEyeImg = img[y1:y2, x1:x2]
    rightEyeImg = cv2.resize(rightEyeImg, (60, 40))

    bothEyeImg = img[y1:y2, lx1:x2]
    bothEyeImg = cv2.resize(bothEyeImg, (140, 40))

    return (originFace, leftEyeImg, rightEyeImg, bothEyeImg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type = str,
        help = "Root of images directory" 
    )
 
    args = parser.parse_args()
    rootdir = args.dir
    eyedir = "Eyes"

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            img_path = rootdir + "\\" + file
            print img_path
            img = cv2.imread(img_path)
            eyeImages = cropEyeImage(img)
            eyeimagePath = eyedir + "\\" + file
            if eyeImages == None:
                continue
            cv2.imwrite(eyeimagePath, eyeImages[3])
            rvec = estimatePose(eyeImages[0], img)
            pickleFile = eyedir + "\\" + file[:-3] + "pickle"
            print pickleFile
            info = {"rot": rvec}
            pickle.dump(info, open(pickleFile, "wb"))
            out = pickle.load(open(pickleFile, "rb"))
            print out["rot"]

