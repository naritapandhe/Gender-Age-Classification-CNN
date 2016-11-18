import pandas as pd
import numpy as np
import PIL
from PIL import Image
import pickle
import random
import matplotlib.pyplot as plt
import json


def save_obj(obj,name):
	image_path = '/Volumes/Mac-B/faces-recognition/new_dicts/aligned/'
	with open(image_path+name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)

def create_train_data():
	print ('Creating train data...')
	fold_names = ['fold_3_data']
	#,'fold_1_data','fold_2_data','fold_3_data']
	i = 0;
	width = 256
	height = 256
	new_width = 227
	new_height = 227

	for fold in fold_names:
		df = pd.read_csv('/Volumes/Mac-B/faces-recognition/csvs/'+fold+'.csv')
		inputimages = []
		genders = []
		print ("Tota no. of rows in %s" % fold)
		print df.shape
		'''
		for index, row in df.iterrows():
			yaw_angle = row['fiducial_yaw_angle']
			gender = row['gender']
			if ((yaw_angle >= -15) and (yaw_angle <= 15) and (gender!='u')):
				if gender == 'f' or gender=='m':
					folder_name = row['user_id']
					image_name = row['original_image']
					face_id = row['face_id']
				
					image_path = '/Volumes/Mac-B/faces-recognition/aligned/'+folder_name+'/landmark_aligned_face.'+str(face_id)+'.'+image_name
					image = Image.open(image_path)
			
					#Resize image
					image = image.resize((width, height), PIL.Image.ANTIALIAS)

					image_arr = np.array(image)
					inputimages.append(image_arr)
					genders.append(gender)
		
		print('Done: {0}/{1} folds'.format(i, len(fold_names)))
		i += 1

		print (fold)            
		print len(inputimages)            
		print len(genders)
		print ('')

		dict = {'fold_name': fold, 'images': inputimages, 'labels': genders}
		#save_obj(dict,fold)
		'''
	
def create_test_data():
	print ('Creating test data...')
	fold_names = ['fold_4_data']
	test_data = []
	i = 0;
	width = 256
	height = 256
	new_width = 227
	new_height = 227

	for fold in fold_names:
		df = pd.read_csv('/Volumes/Mac-B/faces-recognition/csvs/'+fold+'.csv')
		inputimages = []
		genders = []
		folder_names = []
		image_names = []
		face_ids = []
		for index, row in df.iterrows():
			yaw_angle = row['fiducial_yaw_angle']
			gender = row['gender']
			if ((yaw_angle >= -15) and (yaw_angle <= 15) and (gender!='u')):
				if gender == 'f' or gender=='m':
					folder_name = row['user_id']
					image_name = row['original_image']
					face_id = row['face_id']
				
					image_path = '/Volumes/Mac-B/faces-recognition/aligned/'+folder_name+'/landmark_aligned_face.'+str(face_id)+'.'+image_name
					image = Image.open(image_path)
			
					#Resize image
					image = image.resize((width, height), PIL.Image.ANTIALIAS)

					folder_names.append(folder_name)
					image_names.append(image_name)
					face_ids.append(face_id)

					image_arr = np.array(image)
					inputimages.append(image_arr)
					genders.append(gender)

					dict = {'folder_name': folder_name, 'image_name': image_name, 'face_id': face_id, 'gender': gender}
					test_data.append(dict)

		
		print('Done: {0}/{1} folds'.format(i, len(fold_names)))
		i += 1

		print (fold)            
		print len(inputimages)            
		print len(genders)
		print ('')

		dict = {'fold_name': fold, 'images': inputimages, 'labels': genders, 'folder_names': folder_names, 'image_names':image_names,'face_ids':face_ids}
		save_obj(dict,fold)


	print('Writing the testing data to a file....')		
	with open('testing_data.json', 'w') as fout:
		json.dump(test_data, fout)	

if __name__ == '__main__':
	create_train_data()
	#create_test_data()


