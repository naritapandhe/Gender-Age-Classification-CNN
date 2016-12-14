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
from collections import Counter

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

def get_data_stats(fold_names):
	print ('Creating train data...')
	i = 0;
	width = 256
	height = 256
	
	#fold_ages_cnt = [None] * 8

	for fold in fold_names:
		fold_genders_cnt = {}
		fold_genders_cnt[0] = 0
		fold_genders_cnt[1] = 0

		fold_age_cnt = {}
		fold_age_cnt[0] = 0
		fold_age_cnt[1] = 0
		fold_age_cnt[2] = 0
		fold_age_cnt[3] = 0
		fold_age_cnt[4] = 0
		fold_age_cnt[5] = 0
		fold_age_cnt[6] = 0
		fold_age_cnt[7] = 0

		df = pd.read_csv('/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/data/csvs/'+fold+'.csv')
		
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
					
					
						
					if gender == 'm':
						fold_genders_cnt[0]+=1
					else:
						fold_genders_cnt[1]+=1

					fold_age_cnt[age_id]+=1
							
		
		print('Done: {0}/{1} folds'.format(i, len(fold_names)))
		i=i+1
		
		print fold_genders_cnt

		print ('Fold Name: %s' % fold)            
		print ('M: %i, F: %i' % (fold_genders_cnt[0],fold_genders_cnt[1]))
		print (fold_age_cnt)
		#3print (df.count())
		print ('\n')
	
		#print('Done: {%s}/{%s} fold\n' % (fold, fold_names))


def get_age_wise_male_female(fold_names):
	print ('Creating train data...')
	i = 0;
	width = 256
	height = 256
	
	#fold_ages_cnt = [None] * 8
	fold_age0_cnt = {0:0, 1:0}
	fold_age1_cnt = {0:0, 1:0}
	fold_age2_cnt = {0:0, 1:0}
	fold_age3_cnt = {0:0, 1:0}
	fold_age4_cnt = {0:0, 1:0}
	fold_age5_cnt = {0:0, 1:0}
	fold_age6_cnt = {0:0, 1:0}
	fold_age7_cnt = {0:0, 1:0}

	for fold in fold_names:
		fold_genders_cnt = {}
		fold_genders_cnt[0] = 0
		fold_genders_cnt[1] = 0

		


		df = pd.read_csv('/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/data/csvs/'+fold+'.csv')
		
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
					
					if (age_id==0):
						if gender == 'm':
							fold_age0_cnt[0]+=1
						else:
							fold_age0_cnt[1]+=1

					elif (age_id==1): 
						if gender == 'm':
							fold_age1_cnt[0]+=1
						else:
							fold_age1_cnt[1]+=1
					elif (age_id==2): 
						if gender == 'm':
							fold_age2_cnt[0]+=1
						else:
							fold_age2_cnt[1]+=1
					elif (age_id==3): 
						if gender == 'm':
							fold_age3_cnt[0]+=1
						else:
							fold_age3_cnt[1]+=1
					elif (age_id==4): 
						if gender == 'm':
							fold_age4_cnt[0]+=1
						else:
							fold_age4_cnt[1]+=1
					elif (age_id==5): 
						if gender == 'm':
							fold_age5_cnt[0]+=1
						else:
							fold_age5_cnt[1]+=1
					elif (age_id==6): 
						if gender == 'm':
							fold_age6_cnt[0]+=1
						else:
							fold_age6_cnt[1]+=1
					elif (age_id==7): 
						if gender == 'm':
							fold_age7_cnt[0]+=1
						else:
							fold_age7_cnt[1]+=1

							
	
	print ("0:2")
	print(fold_age0_cnt)
	print ('\n')
	
	print ("4:6")
	print(fold_age1_cnt)
	print ('\n')
	
	print ("8:13")
	print(fold_age2_cnt)
	print ('\n')
	
	print ("15:20")
	print(fold_age3_cnt)
	print ('\n')
	
	print ("25:32")
	print(fold_age4_cnt)
	print ('\n')

	print ("38:43")
	print(fold_age5_cnt)
	print ('\n')
	
	print ("48:53")
	print(fold_age6_cnt)
	print ('\n')

	print ("60:100")
	print(fold_age7_cnt)
	print ('\n')


def load_train_file(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)

def get_gender_based_age_data_stats(gender,fold_names,pickle_file_path_prefix):

	'''
	Read every fold based on the gender
	Read the values for age and print the
	counts for each age category
	'''

	for fold in fold_names:
		train_ages = []
	
		print('Trying to read training fold: %s......' % fold)
		train_file = load_obj(pickle_file_path_prefix+fold)

		current_file = train_file
		print current_file['test_data_for']
		ages = np.array(current_file['gt_ages'])
		print ages.shape
		x = Counter(ages)
		print sorted(x.items(), key=lambda i: i[0])
		
		print ("\n")
		

def get_gender_stats(file_names,pickle_file_path_prefix):
	'''
	Read every gender_neutral fold
	Count the no. of males and females in it
	'''
	for file in file_names:
		file_data = load_obj(pickle_file_path_prefix+file)

		print file_data['fold_name']
		genders = np.array(file_data['genders'])
		x = Counter(genders)
		print sorted(x.items(), key=lambda i: i[0])
		
		print ("\n")



def main():
	#train_fold_names = ['fold_0_data','fold_1_data','fold_2_data','fold_3_data','fold_4_data']
	#train_fold_names = ['fold_4_data']
	
	#get_male_data_stats()


	fold_names = ['female_test']
	pickle_file_path_prefix = '/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/gender_based_data/female/'
	get_gender_based_age_data_stats('female',fold_names,pickle_file_path_prefix)

	'''
	file_names = ['fold_0_data','fold_1_data','fold_2_data','fold_3_data','fold_4_data']
	pickle_file_path_prefix = '/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/gender_neutral_data/'
	get_gender_stats(file_names,pickle_file_path_prefix)
	'''

	
if __name__ == "__main__":
	main()

