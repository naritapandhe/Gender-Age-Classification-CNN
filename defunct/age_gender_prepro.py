import pandas as pd
import numpy as np
import PIL
from PIL import Image
import pickle
import random
import matplotlib.pyplot as plt
import json
from ast import literal_eval as make_tuple
from operator import itemgetter


def save_obj(obj,name):
	image_path = '/Volumes/Mac-B/faces-recognition/new_model_dicts/alldata/'
	with open(image_path+name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)

def get_age_range_id(age_tuple):
	age_ranges = [(0,2),(4,6),(8,13),(15,20),(25,32),(38,43),(48,53),(60,100)]
	diff_tuple = []

	if age_tuple:
		for r in age_ranges:
			x = tuple(np.subtract(r,age_tuple))
			x = tuple(np.absolute(x))
			diff_tuple.append(x)
	else:
		x = (-1,0)
		diff_tuple.append(x)


	min_index = diff_tuple.index(min(diff_tuple, key=itemgetter(1)))
	return min_index	


def create_train_data(fold_names):
	print ('Creating train data...')
	i = 0;
	width = 256
	height = 256
	
	for fold in fold_names:
		df = pd.read_csv('/Volumes/Mac-B/faces-recognition/csvs/'+fold+'.csv')
		inputimages = []
		genders = []
		ages = []
		for index, row in df.iterrows():
			yaw_angle = row['fiducial_yaw_angle']
			gender = row['gender']
			age = row['age']
					
			#(yaw_angle >= -45) and (yaw_angle <= 45) and 		
			#if ((gender!='u') and (gender!='Nan')):
				#if gender == 'f' or gender=='m':
					
			folder_name = row['user_id']
			image_name = row['original_image']
			face_id = row['face_id']
			
			age_tuple = make_tuple(age)
			age_id = get_age_range_id(age_tuple)
			
			image_path = '/Volumes/Mac-B/faces-recognition/aligned/'+folder_name+'/landmark_aligned_face.'+str(face_id)+'.'+image_name
			image = Image.open(image_path)
	
			#Resize image
			image = image.resize((width, height), PIL.Image.ANTIALIAS)

			image_arr = np.array(image)
			inputimages.append(image_arr)
			genders.append(gender)
			ages.append(age_id)
		
		print('Done: {0}/{1} folds'.format(i, len(fold_names)))
		i=i+1
		

		print (fold)            
		print len(inputimages)            
		print len(genders)
		print len(ages)
		print ('')

		dict = {'fold_name': fold, 'images': inputimages, 'genders': genders, 'ages': ages}
		save_obj(dict,fold)
		print('Done: {%s}/{%s} fold' % (fold, fold_names))
		
		
	
def create_test_data(fold_names):
	print ('Creating test data...')
	test_data = []
	i = 0;
	width = 256
	height = 256

	for fold in fold_names:
		df = pd.read_csv('/Volumes/Mac-B/faces-recognition/csvs/'+fold+'.csv')
		inputimages = []
		genders = []
		folder_names = []
		image_names = []
		face_ids = []
		ages = []

		for index, row in df.iterrows():
			yaw_angle = row['fiducial_yaw_angle']
			gender = row['gender']
			age = row['age']
				
			#(yaw_angle >= -45) and (yaw_angle <= 45) and 	
			if ((gender!='u') and (gender!='Nan')):
				if gender == 'f' or gender=='m':
					folder_name = row['user_id']
					image_name = row['original_image']
					face_id = row['face_id']
					
					image_path = '/Volumes/Mac-B/faces-recognition/aligned/'+folder_name+'/landmark_aligned_face.'+str(face_id)+'.'+image_name
					image = Image.open(image_path)
			
					#Resize image
					image = image.resize((width, height), PIL.Image.ANTIALIAS)

					age_tuple = make_tuple(age)
					age_id = get_age_range_id(age_tuple)
					
					folder_names.append(folder_name)
					image_names.append(image_name)
					face_ids.append(face_id)

					image_arr = np.array(image)
					inputimages.append(image_arr)
					genders.append(gender)
					ages.append(age_id)

					dict = {'folder_name': folder_name, 'image_name': image_name, 'face_id': face_id, 'gender': gender, 'age':age_id}
					test_data.append(dict)

		
		print('Done: {0}/{1} folds'.format(i, len(fold_names)))
		i += 1

		print (fold)            
		print len(inputimages)            
		print len(genders)
		print len(ages)
		print ('')

		dict = {'fold_name': fold, 'images': inputimages, 'genders': genders,'ages':ages, 'folder_names': folder_names, 'image_names':image_names,'face_ids':face_ids}
		save_obj(dict,fold)


	print('Writing the testing data to a file....')		
	with open('/Volumes/Mac-B/faces-recognition/new_model_dicts/alldata/testing_data.json', 'w') as fout:
		json.dump(test_data, fout)	

if __name__ == '__main__':
	train_fold_names = ['fold_0_data','fold_1_data','fold_2_data','fold_3_data']
	create_train_data(train_fold_names)
	
	#test_fold_names = ['fold_4_data']
	#create_test_data(test_fold_names)


