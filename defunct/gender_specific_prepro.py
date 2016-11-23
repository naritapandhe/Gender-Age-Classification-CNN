import pandas as pd
import numpy as np
import PIL
from PIL import Image
import pickle
import json
from ast import literal_eval as make_tuple
from operator import itemgetter
import sys


def save_obj(obj,name, save_path):
	with open(save_path+name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def get_age_range_id(age_tuple):
	age_ranges = [(0,2),(4,6),(8,13),(15,20),(25,32),(38,43),(48,53),(60,100)]
	diff_tuple = []

	for r in age_ranges:
		x = tuple(np.subtract(r,age_tuple))
		x = tuple(np.absolute(x))
		diff_tuple.append(x)

	min_index = diff_tuple.index(min(diff_tuple, key=itemgetter(1)))
	return min_index	


def extract_gender_train_data(gen, fold_names, save_path):
	print ('Extracting data for: %s' % gen)
	i = 0;
	width = 256
	height = 256

	for fold in fold_names:
		df = pd.read_csv('/Volumes/Mac-B/faces-recognition/csvs/'+fold+'.csv')
		inputimages = []
		ages = []
		genders = []
		for index, row in df.iterrows():
			yaw_angle = row['fiducial_yaw_angle']
			gender = row['gender']
			age = row['age']
					
			if ((yaw_angle >= -15) and (yaw_angle <= 15) and (gender!='u') and (gender!='Nan') and (age!='None')):
				if gender == gen:
					
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
					ages.append(age_id)
					genders.append(gender)
		
		print('Done: {0}/{1} folds'.format(i, len(fold_names)))
		i=i+1
		

		print (fold)            
		print len(inputimages)            
		print len(genders)
		print len(ages)
		print ('')

		dict = {'fold_name': fold, 'images': inputimages, 'genders': genders, 'ages': ages}
		save_obj(dict,fold,save_path)
		print('Done: {%s}/{%s} fold' % (fold, fold_names))
	
	
def extract_gender_test_data(gen,fold_names,save_path):
	print ('Creating test data for: %s' % gen)
	test_data = []
	i = 0;
	width = 256
	height = 256

	for fold in fold_names:
		df = pd.read_csv('/Volumes/Mac-B/faces-recognition/csvs/'+fold+'.csv')
		inputimages = []
		folder_names = []
		image_names = []
		face_ids = []
		ages = []
		genders = []

		for index, row in df.iterrows():
			yaw_angle = row['fiducial_yaw_angle']
			gender = row['gender']
			age = row['age']
				
			if ((yaw_angle >= -15) and (yaw_angle <= 15) and (gender!='u') and (gender!='Nan') and (age!='None')):
				if gender == gen:
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
					ages.append(age_id)
					genders.append(gender)

					dict = {'folder_name': folder_name, 'image_name': image_name, 'face_id': face_id, 'actual_gender': gender, 'actual_age':age_id}
					test_data.append(dict)

		
		print('Done: {0}/{1} folds'.format(i, len(fold_names)))
		i += 1

		print (fold)            
		print len(inputimages)            
		print len(ages)
		print len(genders)
		print ('')

		dict = {'fold_name': fold, 'images': inputimages, 'genders': genders,'ages':ages, 'folder_names': folder_names, 'image_names':image_names,'face_ids':face_ids}
		save_obj(dict,fold,save_path)

	
	print('Writing the testing data to a file....')		
	with open('/Volumes/Mac-B/faces-recognition/new_model_dicts/'+gen+'/testing_data_'+gen+'.json', 'w') as fout:
		json.dump(test_data, fout)	


def main():
	'''
		1. Read the gender for which data is to be extracted
		2. Based on the gender names, define the fold save path
		3. Call the extract functions with respective arguements
	'''
	gender = sys.argv[1]
	print ('Data processing for: %s' % gender)

	if (gender=='m'):
		train_fold_names = ['fold_0_data','fold_1_data','fold_2_data','fold_3_data']
		save_path = '/Volumes/Mac-B/faces-recognition/new_model_dicts/m/'
		extract_gender_train_data('m',train_fold_names,save_path)

		test_fold_names = ['fold_4_data']
		extract_gender_test_data('m',test_fold_names,save_path)
	
	elif (gender=='f'):
		train_fold_names = ['fold_0_data','fold_1_data','fold_2_data','fold_3_data']
		save_path = '/Volumes/Mac-B/faces-recognition/new_model_dicts/f/'
		extract_gender_train_data('f',train_fold_names,save_path)

		test_fold_names = ['fold_4_data']
		extract_gender_test_data('f',test_fold_names,save_path)
	

if __name__	==	'__main__':
	main()

