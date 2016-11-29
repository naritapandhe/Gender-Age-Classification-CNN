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
from pprint import pprint
from skimage import exposure

def save_pickle(obj,name,save_path):
	with open(save_path+name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(path,name):
	with open(path+name + '.pkl', 'rb') as f:
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

def read_fold(fold_csv_prefix,fold_image_prefix,fold_names):
	width = 256
	height = 256
	i=0
	
	for fold in fold_names:
		print("Reading fold: %s" % fold)
		df = pd.read_csv(fold_csv_prefix+fold+'.csv')
		inputimages = []
		genders = []
		ages = []

		
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
					
					image_path = fold_image_prefix+folder_name+'/landmark_aligned_face.'+str(face_id)+'.'+image_name
					image = Image.open(image_path)
			
					#Resize image
					image = image.resize((width, height), PIL.Image.ANTIALIAS)
					image_arr = np.array(image)
					#image_arr = exposure.equalize_hist(image_arr)
					

					if(gender == 'm'):
						g=0
					else:
						g=1
					
					inputimages.append(image_arr)
					genders.append(g)
					ages.append(age_id)
		
		print('Done: {0}/{1} folds'.format(i, len(fold_names)))
		i=i+1

		print ('Fold Name: %s' % fold)            
		print ('Images: %i, Gender: %i, Ages: %i' % (len(inputimages), len(genders), len(ages)))            
		print ('')

		currDict = {'fold_name': fold, 'images': inputimages, 'genders': genders, 'ages': ages}
		save_pickle(currDict,fold, '/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/gender_neutral_data/')


def create_crossval_data(fold_names):

	total_files = len(fold_names)
	cv_train = []
	cv_val=[]

	for i in range(total_files):
		
		intermediate = []
		if(i == 0):
			j = i+3
		else:	 
			j = (i+3)%total_files
		
		k = (i+2)%total_files
		if k < i:
			for ii in range(i,total_files):
				intermediate.append(fold_names[ii])

			for ii in range(k):
				intermediate.append(fold_names[ii])	
			intermediate.append(fold_names[k])	
		else:	
			for ii in range(i,k+1):
				intermediate.append(fold_names[ii])

		'''
		Over here we have all the names of files which should go in train 
		and validation set respectively.
		So, read those files and create the train and validation pickles
		Call the pickle files as: 
			gender_neutral_cv0, gender_neutral_cv1, gender_neutral_cv2, gender_neutral_cv3
		'''		
		cv_train.append(intermediate)
		cv_val.append(fold_names[j])
	
	pprint(cv_train)
	pprint(cv_val)	
	
	for i in range(len(cv_train)):
		train_files = cv_train[i]
		train_data = []
		val_data = []

		for ii in range(len(train_files)):
			train_file_data = load_pickle('/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/gender_neutral_data/',train_files[ii])
			train_data.append(train_file_data)

		val_data = load_pickle('/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/gender_neutral_data/',cv_val[i])	
		print (cv_val[i])
		print len(val_data)

		save_pickle(train_data,'gender_neutral_cv_train_'+str(i), '/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/gender_neutral_data/')
		save_pickle(val_data,'gender_neutral_cv_val_'+str(i), '/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/gender_neutral_data/')

		print("Train and val created for fold: %i" % i)
	
	
def main():
	#fold_names = ['fold_0_data','fold_1_data','fold_2_data','fold_3_data','fold_4_data']	
	#read_fold('/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/data/csvs/','/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/data/aligned/',fold_names)
	
	fold_names = ['fold_0_data','fold_1_data','fold_2_data','fold_3_data']	
	create_crossval_data(fold_names)

if __name__ == "__main__":
	main()


