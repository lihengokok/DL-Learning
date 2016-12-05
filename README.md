How to use the Eye Contact Pipeline

1. In DataCollection/EyeContact folder

a. Use getEyeContactOnFrames.m to generate a xx.csv file using the elan annotation RA111_sn.eaf file for video RA111.mp4

b. Use grabVideoByFrame.py to read the RA111_sn.csv file to grab the annotated frames, you can get a list of images named as 1111_T.png, where T means eye contact, F means no eye contact

2. In Openface folder

a. Use facedetect.py to read the images from step 1. Then you can get a folder of images and pickles files like 1111_T.png and 1111_T.pickle

note: pickle file saves ['rot'] : [1, 2, 3] as a rotation vector
1111_T.png is the cropped eye area image

note: it uses multiprocessor, so the command line will look messy

b. Use prepareHDF5.py to generate randomly mixed training data

Each entry in HDF5 is like this:

['data'] -> 140 X 40 image rotated and flipped by third axe to 1 X 40 X 140 normalized image

['label'] -> [boolean(Eyecontact), double(head pose theta), double(head pose phi)]

Refer to https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild-mpiigaze/ if you need more context about training

3. Caffetraining folder

Note: a,b are optional
a. configure trainEyeContact.prototxt to change the CNN structure
b. configure solverEyeContact.prototxt to change the training configurations: such as iterations and CPU or GPU mode 

c. configure the train_list.txt to selected the hdf5 files as training group

d. run eyeContactTrain.py to train on the dataset

e. Don't wait in front of the screen, go streching!

