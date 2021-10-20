# ES_FaceMatch_Challenge

How to use:-

Download the dlib model from website:- https://github.com/davisking/dlib-models and unzip it. We have used the shape_predictor_5_face_landmarks.dat model for detection of eye features of the face. 

run.py script expect to you to give a csv file which would two columns with the paths to images.

The pipeline for your system is as follow:-

a. Face Alignment  
b. Face extraction  
c. Feature extraction from the cropped face  
d. Using a learned model for final prediction  
