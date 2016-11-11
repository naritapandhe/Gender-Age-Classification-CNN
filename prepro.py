import pandas as pd
import numpy as np
import PIL
from PIL import Image
import pickle

def save_obj(obj,name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

fold_names = ['fold_0_data']
i = 0;
width = 256
height = 256
new_width = 227
new_height = 227

for fold in fold_names:
    df = pd.read_csv('/Volumes/Mac-B/faces-recognition/'+fold+'.csv')
    inputimages = []
    genders = []
    for index, row in df.iterrows():
        yaw_angle = row['fiducial_yaw_angle']
        if ((yaw_angle >= -15) and (yaw_angle <= 15)):
            folder_name = row['user_id']
            image_name = row['original_image']
            face_id = row['face_id']
            gender = row['gender']
            image_path = '/Volumes/Mac-B/faces-recognition/aligned/'+folder_name+'/landmark_aligned_face.'+str(face_id)+'.'+image_name
            image = Image.open(image_path)
        
            #Resize image
            image = image.resize((256, 256), PIL.Image.ANTIALIAS)
        
            #Center Crop
            '''
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2
            image = image.crop((left, top, right, bottom))
            '''

            image_arr = np.array(image)
            inputimages.append(image_arr)
            genders.append(gender)
            print i
            i=i+1


    dict = {'fold_name': fold, 'images': inputimages, 'labels': genders}
    save_obj(dict,fold)
    
        

