import cv2
from facenet_pytorch import MTCNN
import torch
import pickle
import argparse

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from imageio import imread
from scipy.spatial import distance
import torch 
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

import pandas as pd
from tqdm import tqdm
import dlib
from align import AlignDlib
import glob
import mediapipe as mp



# PRE-PROCESSING
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def align_face(face):
    (h,w,c) = face.shape
    bb = dlib.rectangle(0, 0, w, h)
    aligned_face, npLandmarks = alignment.align(96, face, bb,landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    return aligned_face, npLandmarks
  
def load_and_align_images(filepaths):
    aligned_images = []
    for filepath in filepaths:
        #print(filepath)
        img = cv2.imread(filepath)
        aligned = align_face(img)
        aligned = (aligned / 255.).astype(np.float32)
        aligned = aligned.transpose((2, 0, 1))
        aligned = np.expand_dims(aligned, axis=0)
        aligned_images.append(aligned)
            
    return np.array(aligned_images)
    
def calc_embs(filepaths, batch_size=2): 
    pd = []
    for start in tqdm(range(0, len(filepaths), batch_size)):
        aligned_images = load_and_align_images(filepaths[start:start+batch_size])
        aligned_images = torch.from_numpy(aligned_images).squeeze(1)
        emb = nn4_small2(aligned_images).detach().numpy()
        pd.append(emb)
    #embs = l2_normalize(np.concatenate(pd))
    embs = np.concatenate(pd, axis=0)

    return embs
    
def align_faces(faces):
    aligned_images = []
    landmarks_of_face_list = []
    for face in faces:
        #print(face.shape)
        aligned, npLandmarks = align_face(face)
        aligned = (aligned / 255.).astype(np.float32)
        aligned = aligned.transpose((2, 0, 1))
        aligned_images.append(aligned)
        landmarks_of_face_list.append(npLandmarks)
        
    return aligned_images, landmarks_of_face_list

def calc_emb_test(faces):
    pd = []
    aligned_faces, landmarks_of_face_list = align_faces(faces)
    aligned_faces = np.array(aligned_faces)
    aligned_faces = torch.from_numpy(aligned_faces)
    pd.append(nn4_small2(aligned_faces).detach().numpy())
    #embs = l2_normalize(np.concatenate(pd))
    embs = np.concatenate(pd, axis=0)
    return np.array(embs), landmarks_of_face_list

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img)

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

def mtcnn_detect(image):
    boxes, _ = mtcnn.detect(frame)
    faces = []
    if boxes is not None:
        for box in boxes:
            bbox = list(map(int,box.tolist()))
            faces.append(frame[bbox[1] : bbox[3], bbox[0] : bbox[2]])
    return boxes, faces

def hog_svm_detect(image):
    faceRects = hogFaceDetector(image, 0)
    boxes = []
    faces = []
    
    for faceRect in faceRects:
        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()
        faces.append(image[y1 : y2, x1 : x2])
        boxes.append(np.array([x1, y1, x2, y2]))
    return boxes, faces

# INITIALIZE MODELS
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

nn4_small2 = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(device)

nn4_small2.eval()

alignment = AlignDlib('weights/shape_predictor_68_face_landmarks.dat')

mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)
hogFaceDetector = dlib.get_frontal_face_detector()

#LOAD TRAINING INFORMATION
train_paths = glob.glob("image/*")
print(train_paths)

nb_classes = len(train_paths)

df_train = pd.DataFrame(columns=['image', 'label', 'name'])

for i,train_path in enumerate(train_paths):
    name = os.path.basename(train_path)
    images = glob.glob(train_path + "/*")
    for image in images:
        df_train.loc[len(df_train)]=[image,i,name]
        
print(df_train.head())


# TRAINING
label2idx = []

for i in tqdm(range(len(train_paths))):
    label2idx.append(np.asarray(df_train[df_train.label == i].index))
    
train_embs = np.load("train_embs.npy")
threshold = 1


model_dict = pickle.load(open('./weights/model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Fuck', 1: 'Okay', 2: 'Yeah'}


cap = cv2.VideoCapture(0)
while cap.isOpened():
    data_aux = []
    x_ = []
    y_ = []
    isSuccess, frame = cap.read()
    vis_frame = frame.copy()
    if isSuccess:
        # Hand Detection
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    vis_frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(vis_frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
        
        # Face Detection
        boxes, faces = hog_svm_detect(frame)
        if not faces:
            print("no face detected!")
            continue
        else:
            print("Face Detected")
            try: 
                test_embs, lm_per_face_list = calc_emb_test(faces)
            except:
                continue
            
        people = []
        for i in range(test_embs.shape[0]):
            distances = []
            for j in range(len(train_paths)):
                distances.append(np.min([distance.euclidean(test_embs[i].reshape(-1), train_embs[k].reshape(-1)) for k in label2idx[j]]))
                
            if np.min(distances)>threshold:
                people.append("unknown")
            else:
                res = np.argsort(distances)[:1]
                people.append(res)
                
        names = []
        title = ""
        for p in people:
            if p == "unknown":
                name = "unknown"
            else:
                name = df_train[(df_train['label']==p[0])].name.iloc[0]
                name = name.split("/")[-1]
            names.append(name)
            title = title + name + " "
        
        for i, faceRect in enumerate(boxes):
            faceRect = list(map(int,faceRect.tolist()))
            x1 = faceRect[0]
            y1 = faceRect[1]
            x2 = faceRect[2]
            y2 = faceRect[3]
            npLandmarks = lm_per_face_list[i]
            npLandmarks = npLandmarks + np.array([[x1, y1]])
            for landmark in npLandmarks:
                landmark = landmark.tolist()
                vis_frame = cv2.circle(vis_frame, (int(landmark[0]), int(landmark[1])), radius=1, color=(0, 0, 255), thickness=2)
            cv2.rectangle(vis_frame,(x1,y1),(x2,y2),(255,0,0),3)
            cv2.putText(vis_frame,names[i],(x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),1,cv2.LINE_AA)
            
                
    cv2.imshow('Face Detection', vis_frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
