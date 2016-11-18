import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
import sys
import pickle
import random


def one_hot(y):
    y_ret = np.zeros((len(y), 2))
    y_ret[np.arange(len(y)), y.astype(int)] = 1
    return y_ret

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


#Read the data from pickles
pickle_file_paths = ['fold_0_data','fold_1_data','fold_2_data','fold_3_data','fold_4_data']
#['fold_3_data']
#['fold_0_data','fold_1_data','fold_2_data','fold_3_data','fold_4_data']
pickle_file_path_prefix = '/Volumes/Mac-B/faces-recognition/new_dicts/aligned/'
#pickle_file_path_prefix = '/home/ubuntu/gender_age/data/'

X = []
y = []
X_test = []
folder_names = []
image_names = []
face_ids = []



for pf in pickle_file_paths:
    pfile = load_obj(pickle_file_path_prefix+pf)
    images = []
    labels = []

    if pf == 'fold_4_data':
        images = (pfile['images'])
        folders_names1 = (pfile['folder_names']) 
        images_names1 = (pfile['image_names'])
        faces_ids1 = (pfile['image_names'])
        
        images = np.array(images)
        X_test.append(images)

        folders_names1 = np.array(folders_names1)
        folder_names.append(folders_names1)

        images_names1 = np.array(images_names1)
        image_names.append(images_names1)

        faces_ids1 = np.array(faces_ids1) 
        face_ids.append(faces_ids1)


    else:

        #dict = {'fold_name': fold, 'images': inputimages, 'labels': genders}
        images = (pfile['images'])
        genders = (pfile['genders'])
        ages = (pfile['ages'])

        images = np.array(images)
        genders = np.array(genders)
        ages = np.array(ages)
        
        indices = np.where(labels =='nan')
        images = np.delete(images,indices,axis=0)
        genders = np.delete(genders, indices)
        ages = np.delete(ages, indices)

        indices = np.where(y =='u')
        images = np.delete(images,indices,axis=0)
        genders = np.delete(genders, indices)
        ages = np.delete(ages, indices)

        genders[genders == 'm'] = 0
        genders[genders == 'f'] = 1

        genders = one_hot(genders)
        
        '''
            Currently, appending just the genders
            Not ages
        '''
        X.append(images)
        y.append(genders)


print len(X)
print len(y)        

X = np.array(X)
X = np.vstack(X)

y = np.array(y)
y = np.vstack(y)

print X.shape
print y.shape

#print X[0]
print y[0]

if X_test:
    print "In X_test"
    X_test = np.array(X_test)
    X_test = np.vstack(X_test)
    print X_test.shape    

