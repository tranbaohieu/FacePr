import cv2
import glob
import pandas as pd
from imageio import imread,imsave
from skimage.transform import resize
from tqdm import tqdm
import dlib


train_paths = glob.glob("image/me/*")
print(train_paths)

# df_train = pd.DataFrame(columns=['image', 'label', 'name'])

# for i,train_path in tqdm(enumerate(train_paths)):
#     name = train_path.split("/")[-1]
#     images = glob.glob(train_path + "/*")
#     for image in images:
#         df_train.loc[len(df_train)]=[image,i,name]
        
# print(df_train)
        
for img_path in train_paths:
    print(img_path)
    image = cv2.imread(img_path)
    hogFaceDetector = dlib.get_frontal_face_detector()
    faceRects = hogFaceDetector(image, 0)

    # win = dlib.image_window()
    # win.clear_overlay()
    # win.set_image(image)
    # win.add_overlay(faceRects)
    
    for faceRect in faceRects:
        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()

        face = image[y1:y2,x1:x2]
        try: 
            cv2.imwrite(img_path,face)
        except:
            
            continue