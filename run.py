import cv2
import numpy as np
from PIL import Image
import dlib
from mtcnn.mtcnn import MTCNN
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from keras.regularizers import l2
from xgboost import XGBClassifier
from tqdm import tqdm
from keras_facenet import FaceNet
import pickle

detector1 = MTCNN()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('dlib-models/shape_predictor_5_face_landmarks.dat')

#xgboost
file_name = "xgb_reg.pkl"
xgb_model_loaded = pickle.load(open(file_name, "rb"))

#nn
model = Sequential()
model.add(Dense(512,  input_shape = (1024,)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.1))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dense(32))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dense(1, activation= 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights('model2_weights.h5')

def shape_to_normal(shape):
    shape_normal = []
    for i in range(0, 5):
        shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
    return shape_normal

def get_eyes_nose_dlib(shape):
    nose = shape[4][1]
    left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
    left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
    right_eyes_x = int(shape[1][1][0] + shape[0][1][0]) // 2
    right_eyes_y = int(shape[1][1][1] + shape[0][1][1]) // 2
    return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)

def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def cosine_formula(length_line1, length_line2, length_line3):
    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
    return cos_a

def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def is_between(point1, point2, point3, extra_point):
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
    if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
        return True
    else:
        return False

def image_alignment(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if len(rects) > 0:
        for rect in rects:
            x = rect.left()
            y = rect.top()
            w = rect.right()
            h = rect.bottom()
            shape = predictor(gray, rect)
        shape = shape_to_normal(shape)
        nose, left_eye, right_eye = get_eyes_nose_dlib(shape)
        center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        center_pred = (int((x + w) / 2), int((y + y) / 2))
        length_line1 = distance(center_of_forehead, nose)
        length_line2 = distance(center_pred, nose)
        length_line3 = distance(center_pred, center_of_forehead)
        cos_a = cosine_formula(length_line1, length_line2, length_line3)
        angle = np.arccos(cos_a)
        rotated_point = rotate_point(nose, center_of_forehead, angle)
        rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
        if is_between(nose, center_of_forehead, center_pred, rotated_point):
            angle = np.degrees(-angle)
        else:
            angle = np.degrees(angle)
        img = Image.fromarray(img)
        img_aligned = np.array(img.rotate(angle))
        return img_aligned
    else: return img


def face_extract(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector1.detect_faces(img1)
    if len(faces)==0:
        return img
    x, y, width, height = faces[0]['box']
    face_aligned =  img1[y:y+height,x:x+width]
    face_aligned = cv2.cvtColor(face_aligned, cv2.COLOR_RGB2BGR)
    return face_aligned



df = pd.read_csv('path/to/test/csv')
xtest = []
problematic = []
embedder = FaceNet()
for i in tqdm(df.iterrows()):
  try:
      ii = cv2.imread('path/to/data/folder/{}'.format(i[1][0]))
      qq = image_alignment(ii)
      tt = face_extract(qq)
      ii1 = cv2.imread('path/to/data/folder/{}'.format(i[1][1]))
      qq1 = image_alignment(ii1)
      tt1 = face_extract(qq1)
      img = np.reshape(tt, (1,tt.shape[0], -1,3))
      embedding = embedder.embeddings(img)
      img1 = np.reshape(tt1, (1,tt1.shape[0], -1,3))
      embedding1 = embedder.embeddings(img1)
      comb = np.concatenate((embedding,embedding1), axis=-1)
      xtest.append(comb)
  except:
    problematic.append(i[0])

xtest = np.array(xtest)
xtest= xtest.reshape([df.shape[0],1024])
#predictions

#nn
ypred = model.predict(xtest) >0.5

#xgboost
ypred = xgb_model_loaded.predict(xtest) >0.5

df['label_pred'] = ypred
df['label_pred'] = df['label_pred'].astype('int')
df.to_csv('submission.csv',index=False)