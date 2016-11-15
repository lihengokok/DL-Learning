import numpy
import cv2
import os
import sys
from face_landmark_detection import faceLandmarkDetection

#For the frontal face detector
import dlib

#Antropometric constant values of the human head. 
#Found on wikipedia and on:
# "Head-and-Face Anthropometric Survey of U.S. Respirator Users"
#
#X-Y-Z with X pointing forward and Y on the left.
#The X-Y-Z coordinates used are like the standard
# coordinates of ROS (robotic operative system)
P3D_RIGHT_SIDE = numpy.float32([-100.0, -77.5, -5.0]) #0
P3D_GONION_RIGHT = numpy.float32([-110.0, -77.5, -85.0]) #4
P3D_MENTON = numpy.float32([0.0, 0.0, -122.7]) #8
P3D_GONION_LEFT = numpy.float32([-110.0, 77.5, -85.0]) #12
P3D_LEFT_SIDE = numpy.float32([-100.0, 77.5, -5.0]) #16
P3D_FRONTAL_BREADTH_RIGHT = numpy.float32([-20.0, -56.1, 10.0]) #17
P3D_FRONTAL_BREADTH_LEFT = numpy.float32([-20.0, 56.1, 10.0]) #26
P3D_SELLION = numpy.float32([0.0, 0.0, 0.0]) #27
P3D_NOSE = numpy.float32([21.1, 0.0, -48.0]) #30
P3D_SUB_NOSE = numpy.float32([5.0, 0.0, -52.0]) #33
P3D_RIGHT_EYE = numpy.float32([-20.0, -65.5,-5.0]) #36
P3D_RIGHT_TEAR = numpy.float32([-10.0, -40.5,-5.0]) #39
P3D_LEFT_TEAR = numpy.float32([-10.0, 40.5,-5.0]) #42
P3D_LEFT_EYE = numpy.float32([-20.0, 65.5,-5.0]) #45
#P3D_LIP_RIGHT = numpy.float32([-20.0, 65.5,-5.0]) #48
#P3D_LIP_LEFT = numpy.float32([-20.0, 65.5,-5.0]) #54
P3D_STOMION = numpy.float32([10.0, 0.0, -75.0]) #62


#The points to track
#These points are the ones used by PnP
# to estimate the 3D pose of the face
TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
ALL_POINTS = list(range(0,68)) #Used for debug only

cam_w = 1280
cam_h = 720

c_x = cam_w / 2
c_y = cam_h / 2
f_x = c_x / numpy.tan(60/2 * numpy.pi / 180)
f_y = f_x

#Estimated camera matrix values.
camera_matrix = numpy.float32([[f_x, 0.0, c_x],
                               [0.0, f_y, c_y], 
                               [0.0, 0.0, 1.0] ])

print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")

#Distortion coefficients
camera_distortion = numpy.float32([0.0, 0.0, 0.0, 0.0, 0.0])


#This matrix contains the 3D points of the
# 11 landmarks we want to find. It has been
# obtained from antrophometric measurement
# on the human head.
landmarks_3D = numpy.float32([P3D_RIGHT_SIDE,
                              P3D_GONION_RIGHT,
                              P3D_MENTON,
                              P3D_GONION_LEFT,
                              P3D_LEFT_SIDE,
                              P3D_FRONTAL_BREADTH_RIGHT,
                              P3D_FRONTAL_BREADTH_LEFT,
                              P3D_SELLION,
                              P3D_NOSE,
                              P3D_SUB_NOSE,
                              P3D_RIGHT_EYE,
                              P3D_RIGHT_TEAR,
                              P3D_LEFT_TEAR,
                              P3D_LEFT_EYE,
                              P3D_STOMION])

#Declaring the two classifiers
my_face_detector = dlib.get_frontal_face_detector()
my_detector = faceLandmarkDetection('shape_predictor_68_face_landmarks.dat')

#Error counter definition
no_face_counter = 0

#Variables that identify the face
#position in the main frame.
face_x1 = 0
face_y1 = 0
face_x2 = 0
face_y2 = 0
face_w = 0
face_h = 0

#Variables that identify the ROI
#position in the main frame.
roi_x1 = 0
roi_y1 = 0
roi_x2 = cam_w
roi_y2 = cam_h
roi_w = cam_w
roi_h = cam_h
roi_resize_w = int(cam_w/10)
roi_resize_h = int(cam_h/10)


def findFace(frame):
  faces_array = my_face_detector(frame, 1)
  face = None
  lowestY = 0
  for i, pos in enumerate(faces_array):
    y1, y2 = pos.top(), pos.bottom()
    if (y1 + y2) / 2 > lowestY:
        lowestY = (y1 + y2) / 2
        face = pos
  return face

def estimatePose(pos, frame):
  face_x1 = pos.left()
  face_y1 = pos.top()
  face_x2 = pos.right()
  face_y2 = pos.bottom()
  text_x1 = face_x1
  text_y1 = face_y1 - 3
  
  '''
  cv2.putText(frame, "FACE ", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1);
  cv2.rectangle(frame, 
               (face_x1, face_y1), 
               (face_x2, face_y2), 
               (0, 255, 0), 
                2)            
  '''
  landmarks_2D = my_detector.returnLandmarks(frame, face_x1, face_y1, face_x2, face_y2, points_to_return=TRACKED_POINTS)


  for point in landmarks_2D:
      cv2.circle(frame,( point[0], point[1] ), 2, (0,0,255), -1)


  #Applying the PnP solver to find the 3D pose
  # of the head from the 2D position of the
  # landmarks.
  #retval - bool
  #rvec - Output rotation vector that, together with tvec, brings 
  # points from the model coordinate system to the camera coordinate system.
  #tvec - Output translation vector.
  retval, rvec, tvec = cv2.solvePnP(landmarks_3D, 
                                    landmarks_2D, 
                                    camera_matrix, camera_distortion)
  return rvec
  '''
  #Now we project the 3D points into the image plane
  #Creating a 3-axis to be used as reference in the image.
  axis = numpy.float32([[50,0,0], 
                        [0,50,0], 
                        [0,0,50]])

  imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)

  #Drawing the three axis on the image frame.
  #The opencv colors are defined as BGR colors such as: 
  # (a, b, c) >> Blue = a, Green = b and Red = c
  #Our axis/color convention is X=R, Y=G, Z=B
  sellion_xy = (landmarks_2D[7][0], landmarks_2D[7][1])
  cv2.line(frame, sellion_xy, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
  cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()), (255,0,0), 3) #BLUE
  cv2.line(frame, sellion_xy, tuple(imgpts[0].ravel()), (0,0,255), 3) #RED

  #outFileName = imageFileName[:-4] + "headpose.png"
  #print outFileName
  #cv2.imwrite("/output/" + outFileName, frame)
  return rvec
  '''
'''
rootdir = "RA145"
counter = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        img_path = subdir + "\\" +  file
        frame = cv2.imread(img_path)
        face = findFace(frame)
        if face == None:
          continue
        rvec = estimatePose(face, frame)
        print rvec
'''