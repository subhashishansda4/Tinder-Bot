from creds import TINDER_AUTH_TOKEN

import datetime
from time import sleep
from random import random
import os
import pandas as pd
import pickle

import requests

from pred import DETECT
from pred import RATE

from keras_vggface.vggface import VGGFace
from mtcnn import MTCNN
from tqdm import tqdm


TINDER_URL = "https://api.gotinder.com"

class API():
    def __init__(self, token):
        self._token = token
        
    def profile(self):
        data = requests.get(TINDER_URL + "/v2/profile?include=account%2Cuser", headers={'x-auth-token': self._token}).json()
        return PROFILE(data['data'], self)
    
    def search(self):
        data = requests.get(TINDER_URL + "/v2/recs/core", headers={'x-auth-token': self._token}).json()
        return list(map(lambda user: PERSON(user['user'], self), data['data']['results']))
    
    
    def like(self, user_id):
        data = requests.get(TINDER_URL + f"/like/{user_id}", headers={'x-auth-token': self._token}).json()
        return {
            'is_match': data['match'],
            'liked_remaining': data['likes_remaining']
        }
    
    def dislike(self, user_id):
        requests.get(TINDER_URL + f"/pass/{user_id}", headers={'x-auth-token': self._token}).json()
        return True
    
    
    def matches(self, limit=10):
        data = requests.get(TINDER_URL + f"/v2/matches?count={limit}", headers={'x-auth-token': self.token}).json()
        return list(map(lambda match: PERSON(match['person'], self), data['data']['matches']))




PROF_FILE = "A:/O/full stack/PROJECTS/Tinder-Bot/images/profiles.txt"
LIKE_FILE = "A:/O/full stack/PROJECTS/Tinder-Bot/images/likes.txt"
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')
detector = MTCNN()
features = pickle.load(open('embeddings.pkl', 'rb'))

class PERSON(object):
    def __init__(self, data , api):
        self._api = api
        
        self.id = data['_id']
        self.name = data.get('name', 'unknown')
        self.bio = data.get('bio', '')
        self.distance = data.get('distance_mi', 0) / 1.60934
        self.location = data.get('city').get('name') if 'city' in data else ""
        
        self.birth_date = pd.to_datetime(datetime.datetime.strptime(
            data['birth_date'], '%Y-%m-%dT%H:%M:%S.%fZ')).date() if data.get(
                'birth_date', False) else None
        
        '''self.gender = ['Male', 'Female', 'Unknown'][data.get('gender', 2)]'''
        
        self.images = list(map(lambda photo: photo['url'], data.get('photos', [])))
        
        self.jobs = ', '.join([job['title'].get('name') if 'title' in job else "" + job['company'].get('name') if 'company' in job else "" for job in data.get('jobs', [])]) if 'jobs' in data else ""
        self.schools = ', '.join([school['name'] for school in data.get('schools', [])]) if 'schools' in data else ""
    
        self.intent = data.get('relationship_intent').get('body_text') if 'relationship_intent' in data else ""
        
        self.descriptors = '\n'.join([desc['name'] + " - " + desc['choice_selections'][0].get('name') for desc in data.get('selected_descriptors')]) if 'selected_descriptors' in data else ""

        
    def __repr__(self):
        return f"{self.id} - {self.name} ({self.birth_date.strftime('%d.%m.%Y')})"
    
    
    
    
    def bio_data(self):
        '''
        with open(PROF_FILE, 'r') as f:
            lines = f.readlines()
            if self.id in lines:
                return
        '''
        
        with open(PROF_FILE, 'a', encoding='utf-8') as f:
            f.write('\n' + 'user: ' + str(self.id) + '\n' +
                    'name: ' + str(self.name) + '\n' +
                    'bio: ' + str(self.bio) + '\n' +
                    'birhday: ' + str(self.birth_date) + '\n' +
                    'distance: ' + str(self.distance) + '\n' +
                    'location: ' + str(self.location) + '\n' +
                    'jobs: ' + str(self.jobs) + '\n' +
                    'schools: ' + str(self.schools) + '\n' +
                    'intent: ' + str(self.intent) + '\n\n' +
                    'descriptors: ' + '\n' + str(self.descriptors) + '\n\n' +
                    '---------------------------------------------------------' + '\n'
                )
            
    def like_data(self, score, sim):
        with open(LIKE_FILE, 'a', encoding='utf-8') as f:
            f.write('\n' + 'user: ' + str(self.id) + '\n' +
                    'name: ' + str(self.name) + '\n' +
                    'score: ' + str(score) + '\n' +
                    'actress: ' + str(sim) + '\n\n' +
                    '-----------------------------------' + '\n'
                )
            
    def like(self):
        return self._api.like(self.id)
    
    def dislike(self):
        return self._api.dislike(self.id)
    
    
    def download_images(self, folder='A:/O/full stack/PROJECTS/Tinder-Bot/images/', sleep_max_for=0):
        index = 0
        for image_url in self.images:
            index += 1
            req = requests.get(image_url, stream=True)
            if req.status_code == 200:
                dirc = f"{self.name}__________{self.id}"
                path = os.path.join(folder, 'unclassified', dirc)

                
                os.mkdir(path) if os.path.exists(path) is False else None

                
                with open(f"{path}/{index}_{self.name}_{self.id}.jpg", 'wb') as f:
                    f.write(req.content)
            sleep(random()*sleep_max_for)
            
            
    def classify_images(self, folder='A:/O/full stack/PROJECTS/Tinder-Bot/images/'):
        dirc = f"{self.name}__________{self.id}"
        path = os.path.join(folder, 'unclassified', dirc)
        path_ = os.path.join(folder, 'classified', dirc)
                
        os.mkdir(path_) if os.path.exists(path_) is False else None
                
        for file in tqdm(os.listdir(path)):
            x = DETECT(os.path.join(path, file), detector)
            if x is None:
                return None
            else:
                x.save(f"{os.path.join(path_, file)}")


    def rate_images(self, folder='A:/O/full stack/PROJECTS/Tinder-Bot/images/classified/'):
        dirc = f"{self.name}__________{self.id}"
        path = os.path.join(folder, dirc)
        arr = []
        for file in tqdm(os.listdir(path)):
            photo = os.path.join(path, file)
            arr.append(RATE(photo, model, features))
        
        x = sorted(list(enumerate(arr)), reverse=True, key=lambda x:x[1])
        try:
            return [x[0][1][0], x[0][1][1]]
        except:
            return None

    
    

class PROFILE(PERSON):
    def __init__(self, data, api):
        super().__init__(data['user'], api)
        
        self.email = data['account'].get('email')
        self.phone_number = data['account'].get('account_phone_number')
        
        self.age_min = data['user']['age_filter_min']
        self.age_max = data['user']['age_filter_max']
        
        self.max_distance = data['user']['distance_filter']
        self.gender_filter = ['Male', 'Female'][data['user']['gender_filter']]
        
        
        
        
        
if __name__ == "__main__":
    token = TINDER_AUTH_TOKEN
    api = API(token)
    
    while True:
        persons = api.search()
        for person in persons:
            person.bio_data()
            person.download_images()
            person.classify_images()
            score = person.rate_images()
            
            print("")
            print(person.id)
            print(person.name)
            print("")
            
            if score is None:
                person.dislike()
                print('no face detected')
                print('PASS')
            else:
                print(score[0], score[1])
                if score[0]>0.5:
                    person.like()
                    person.like_data(score[0], score[1])
                    print('LIKED')
                    print(person.like())
                else:
                    person.dislike()
                    print('PASS')
        
            print("")
            print("-----------------------------------------------")
            print("")
