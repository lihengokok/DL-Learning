import numpy as np 
import cv2
import csv
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type = str,
    )
    parser.add_argument(
        "--mp4",
        type = str,
    )
    parser.add_argument(
        "--dir",
        type = str,
    )
    # Read frame marker
    args = parser.parse_args()
    contactFrameList = []


    with open(args.csv, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            contactFrameList.append(int(row[0]))

    contactFrameList = sorted(contactFrameList)


    cap = cv2.VideoCapture(args.mp4)
    currentFrame = 1
    startFrame = int(contactFrameList[0])
    endFrame = int(contactFrameList[-1])
    writeAll = False
    count = 0

    # open output folder file
    os.chdir(args.dir)
    print 'Finding Start Frame...'
    while(cap.isOpened()):
        if currentFrame == endFrame + 1:
            break

        if not writeAll:
            cap.grab()
            currentFrame += 1
            if (startFrame - currentFrame) == 1:
                writeAll = True
            continue

        ret, frame = cap.read()
        if writeAll:
            
            if currentFrame in contactFrameList:
                cv2.imwrite(str(count) + "_T.png", frame) 
            else:
                cv2.imwrite(str(count) + "_F.png", frame) 
            count += 1

        currentFrame += 1
        if currentFrame % 100 == 0:
            print 'Processing frame: ', currentFrame

    cap.release()