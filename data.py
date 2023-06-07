from bing_image_downloader import downloader

import os
import pickle

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import cv2
from PIL import Image
from tqdm import tqdm


# ------------------------------------------------------------------------------------------------------------------------------

z = ['Dia Mirza', 'Disha Patani', 'Mrunal Thakur', 'Ileana D Cruz', 'Kiara Advani', 'Diana Penty']
for i in range(len(z)):
    downloader.download(z[i], limit=10, output_dir='A:/O/full stack/PROJECTS/Tinder-Bot/dataset/unclassified/',
                        adult_filter_off=True, force_replace=False, timeout=60, verbose=True)


# ------------------------------------------------------------------------------------------------------------------------------


path = 'A:/O/full stack/PROJECTS/Tinder-Bot/dataset/unclassified/'
path_ = 'A:/O/full stack/PROJECTS/Tinder-Bot/dataset/classified/'
actors = os.listdir(path)
actors_ = os.listdir(path_)

for actor in actors:
    for file in tqdm(os.listdir(os.path.join(path, actor))):
        img = cv2.imread(os.path.join(path, actor, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces)>0:
            for (x, y, w, h) in faces:
                face = img[y:y + h, x:x + w]
                face_ = Image.fromarray(face)
                face_ = face_.resize((224,224))
                
                dirc = f"{actor}"
                os.mkdir(os.path.join(path_, dirc)) if os.path.exists(os.path.join(path_, dirc)) is False else None
                
                face_.save(f"{os.path.join(path_, dirc)}/{file}")


# ------------------------------------------------------------------------------------------------------------------------------


model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')

features = []
for actor in actors_:
    for file in tqdm(os.listdir(os.path.join(path_, actor))):
        img = image.load_img(os.path.join(path_, actor, file), target_size=(224,224))
        img_arr = image.img_to_array(img)
        img_exp = np.expand_dims(img_arr, axis=0)
        img_pro = preprocess_input(img_exp)
        
        features.append([(model.predict(img_pro).flatten()), actor])

print(features)
pickle.dump(features, open('embeddings.pkl', 'wb'))


# ------------------------------------------------------------------------------------------------------------------------------
'''
PROF_FILE = "A:/O/full stack/PROJECTS/Tinder-Bot/images/profiles.txt"

a = [0.6, 'abc']
print(str(a))

with open(PROF_FILE, 'a', encoding='utf-8') as f:
    f.write(str(a[1]))'''