import os
import pickle

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from mtcnn import MTCNN
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import cv2
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------------------------------------------


def RATE(path, model, features):
    img = image.load_img(path, target_size=(224,224))
    face_arr = image.img_to_array(img)
    img_exp = np.expand_dims(face_arr, axis=0)
    img_pro = preprocess_input(img_exp)
    result = model.predict(img_pro).flatten()
    
    similarity = []
    for i in range(len(features)):
        similarity.append([cosine_similarity(result.reshape(1,-1),features[i][0].reshape(1,-1)), features[i][1]])
    
    index_no = sorted(list(enumerate(similarity)), reverse=True, key=lambda x:x[1])[0][1][0][0][0]
    index_name = sorted(list(enumerate(similarity)), reverse=True, key=lambda x:x[1])[0][1][1]
    
    return [index_no, index_name]



def DETECT(path, detector):
    img = cv2.imread(path)
    results = detector.detect_faces(img)
    
    try:
        x, y, width, height = results[0]['box']
        photo = img[y: y+height, x: x+width]
        
        face = Image.fromarray(photo)
        face = face.resize((224,224))
        return face
    except:
        return None


'''
folder = 'A:/O/full stack/PROJECTS/Tinder-Bot/images/classified/'
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')
features = pickle.load(open('embeddings.pkl', 'rb'))

file = os.listdir(folder)

for files in file:
    arr = []
    for photos in tqdm(os.listdir(os.path.join(folder, files))):
        path = os.path.join(folder, files, photos)
        arr.append(RATE(path, model, features))
        
    x = sorted(list(enumerate(arr)), reverse=True, key=lambda x:x[1])
    try:
        print(x[0][1][0], x[0][1][1])
    except:
        print('no face')
        
    print("")
    print("")

'''
