'''
1. Read every fold csv
2. Read that person's gender, age. (Both values must be present)
4. The json for every image must look like: {fold_name, face_id, image_name, gender_id, age_id, image}
5. Save this for every fold
7. Based on the gender, create 2 separate files for train and test
8. Pass it to CNN. Basically, train a model based on gender for age

Statistics to draw on this data:
1. How many male, females are present every fold
2. How many ppl of every age group are present in the male, female dataset
2. Print the confusion matrix for every age category, how well did the classifier predict
3. Take into considertaion exact and one-off accuracy
'''
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

def save_obj(obj,name,save_path):
	with open(save_path+name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def save_json(obj,name,save_path):
	with open(save_path+name + '.json', 'w') as f:
		json.dump(obj,f)

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

	min_index = diff_tuple.index(min(diff_tuple, key=itemgetter(1)))
	return min_index	

def create_train_data(fold_names):
	print ('Creating train data...')
	i = 0;
	width = 256
	height = 256
	
	for fold in fold_names:
		df = pd.read_csv('/Volumes/Mac-B/faces-recognition/csvs/'+fold+'.csv')
		maleinputimages = []
		malegenders = []
		maleages = []

		femaleinputimages = []
		femalegenders = []
		femaleages = []

		for index, row in df.iterrows():
			yaw_angle = row['fiducial_yaw_angle']
			gender = row['gender']
			age = row['age']
					
			if ((gender!='u') and (gender!='Nan') and (age!='None') and (gender!=' ') and (age!=' ') and (yaw_angle >= -45) and (yaw_angle <= 45)):
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
						
					if gender == 'm':
						maleinputimages.append(image_arr)
						malegenders.append(0)
						maleages.append(age_id)
					else:
						femaleinputimages.append(image_arr)
						femalegenders.append(1)
						femaleages.append(age_id)
							
		
		print('Done: {0}/{1} folds'.format(i, len(fold_names)))
		i=i+1
		

		print ('Fold Name: %s' % fold)            
		print ('Male images: %i, Female images: %i' % (len(maleinputimages), len(femaleinputimages)))            
		print ('Male genders: %i, Female genders: %i' %(len(malegenders), len(femalegenders)))            
		print ('Male ages: %i, Female ages: %i' % (len(maleages), len(femaleages)))            
		print ('')

		maledict = {'fold_name': fold, 'images': maleinputimages, 'genders': malegenders, 'ages': maleages}
		save_obj(maledict,fold, '/Volumes/Mac-B/faces-recognition/22Nov/male/')
		
		femaledict = {'fold_name': fold, 'images': femaleinputimages, 'genders': femalegenders, 'ages': femaleages}
		save_obj(femaledict,fold, '/Volumes/Mac-B/faces-recognition/22Nov/female/')
		
	
		print('Done: {%s}/{%s} fold' % (fold, fold_names))

def create_test_data(fold):
	print ("Creating test data.....")

	i = 0;
	imgID = 0;
	width = 256
	height = 256
	
	df = pd.read_csv('/Volumes/Mac-B/faces-recognition/csvs/'+fold+'.csv')
	maledata = []
	femaledata = []
	malejson = []
	femalejson = []


	femaleinputimages = []
	femalegenders = []
	femaleages = []

	for index, row in df.iterrows():
		imgID+=1
		yaw_angle = row['fiducial_yaw_angle']
		gender = row['gender']
		age = row['age']
				
		if ((gender!='u') and (gender!='Nan') and (age!='None') and (gender!=' ') and (age!=' ') and (yaw_angle >= -45) and (yaw_angle <= 45)):
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
				
			if gender == 'm':
				#maleinputimages.append(image_arr)
				#malegenders.append(0)
				#maleages.append(age_id)
				maledict = {'fold_name': fold, 'image': image_arr, 'gender': 0, 'ages': age_id, 'image_ID': imgID}
				maledata.append(maledict)

				maledict = {'fold_name': fold, 'gender': 0, 'ages': age_id, 'image_ID': imgID}
				malejson.append(maledict)
			else:
				#femaleinputimages.append(image_arr)
				#femalegenders.append(1)
				#femaleages.append(age_id)
				femaledict = {'fold_name': fold, 'image': image_arr, 'gender': 1, 'ages': age_id, 'image_ID': imgID}
				femaledata.append(femaledict)

				femaledict = {'fold_name': fold, 'gender': 1, 'ages': age_id, 'image_ID': imgID}
				femalejson.append(femaledict)
			
			
			i=i+1				
		
	print('Done: {0}/{1} '.format(i, len(df.iterrows())))
		
	

	print ('Fold Name: %s' % fold)            
	print ('Male images: %i, Female images: %i'  % (len(maledict), len(femaledict)))            
	print ('')

	save_obj(maledict,fold, '/Volumes/Mac-B/faces-recognition/22Nov/male/')
	save_json(malejson,fold,'/Volumes/Mac-B/faces-recognition/22Nov/jsons/')
	
	save_obj(femaledict,fold, '/Volumes/Mac-B/faces-recognition/22Nov/female/')
	save_json(femalejson,fold,'/Volumes/Mac-B/faces-recognition/22Nov/jsons/')

def main():
	train_fold_names = ['fold_0_data','fold_1_data','fold_2_data','fold_3_data']
	create_train_data(train_fold_names)

	#test_fold_name = 'fold_4_data'
	#create_test_data(test_fold_name)


if __name__ == "__main__":
	main()

