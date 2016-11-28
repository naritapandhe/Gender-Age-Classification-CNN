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
import math

'''
1. Read each fold for each age group
2. Add counts of each age group
3. Divide by no. of folds. Name it X.
4. Distribute X images per fold


Properties in original CSV:
user_id, original_image, face_id, age, gender, fiducial_yaw_angle


New CSV properties:

Eg:
Fold 0: 30M 20F
30M: 5 3 2 10 6 2 1 1
20F: 0 1 2 8  7 1 1 0

New Distribution:
Fold 0
30M: 4 2 2 6 8 4 3 1 
31F: 2 3 3 7 9 4 2 1


New Distribution:
Fold 1
28M: 3 3 2 6 8 3 2 1 
24F: 1 2 2 6 8 4 1 0

Fold 1: 25M 35F
25M: 2 2 2 2 10 5 4 1
35F: 3 4 3 5 10 7 2 1


Age group 0 = 4M, 2F
Age group 1 = 2M, 2F
Age group 2 = 2M, 2F
Age group 3 = 6M, 6F
Age group 4 = 8M, 8F
Age group 5 = 4M, 4F
Age group 6 = 2M, 1F
Age group 7 = 1M, 1F
'''

no_of_folds = 4

def load_file(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)

def load_pickles(name,no_of_age_groups = 8):
	with open(name+'.pkl', 'rb') as f:
		a=pickle.load(f)
		b=pickle.load(f)
		c=pickle.load(f)
		d=pickle.load(f)
		e=pickle.load(f)
		f1=pickle.load(f)
		g=pickle.load(f)
		h=pickle.load(f)

	return a+b+c+d+e+f1+g+h

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


def read_fold_for_each_group():
	fold_names = ['0','1','2','3']	
	male_pickle_file_path_prefix = '/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/gender_based_data/male/'
	female_pickle_file_path_prefix = '/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/gender_based_data/female/'
	age_range_ids = [0,1,2,3,4,5,6,7]
	male_age_group_counters_across_folds = np.zeros(8)
	female_age_group_counters_across_folds = np.zeros(8)


	for fold in fold_names:
		train_ages = []
	
		print('Trying to read training fold: %s......' % fold)
		male_fold_file = load_file(male_pickle_file_path_prefix+'male_fold_'+fold+'_data')
		male_ages = np.array(male_fold_file['ages'])
		male_age_counter = Counter(male_ages)
		male_age_counter_dict = dict(male_age_counter)
		male_age_counter_dict = sorted(male_age_counter_dict.items(), key=lambda i: i[0])
		print male_age_counter_dict
		male_age_counter_dict = dict(male_age_counter_dict)
		for i in range(8):
			if i in male_age_counter_dict:
				male_age_group_counters_across_folds[i] += male_age_counter_dict[i]


		female_fold_file = load_file(female_pickle_file_path_prefix+'female_fold_'+fold+'_data')
		female_ages = np.array(female_fold_file['ages'])
		female_age_counter = Counter(female_ages)
		female_age_counter_dict = dict(female_age_counter)
		female_age_counter_dict = sorted(female_age_counter_dict.items(), key=lambda i: i[0])
		print female_age_counter_dict
		female_age_counter_dict = dict(female_age_counter_dict)
		for i in range(8):
			if i in female_age_counter_dict:
				female_age_group_counters_across_folds[i] += female_age_counter_dict[i]

		print ("\n")

	male_age_group_counters_across_folds = np.array(male_age_group_counters_across_folds)
	male_age_group_counters_across_folds = male_age_group_counters_across_folds/float(no_of_folds)
	male_age_group_counters_across_folds = np.ceil(male_age_group_counters_across_folds)
	print male_age_group_counters_across_folds
	

	female_age_group_counters_across_folds = np.array(female_age_group_counters_across_folds)
	female_age_group_counters_across_folds = female_age_group_counters_across_folds/float(no_of_folds)
	female_age_group_counters_across_folds = np.ceil(female_age_group_counters_across_folds)
	print female_age_group_counters_across_folds	
	return male_age_group_counters_across_folds, female_age_group_counters_across_folds


def distribute_data(male_age_group_counters_across_folds,female_age_group_counters_across_folds):
	print ("\nIn distributed data....")

	fold_names = ['fold_0_data','fold_1_data','fold_2_data','fold_3_data']	
	male_pickle_file_path_prefix = '/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/gender_based_data/male/'
	female_pickle_file_path_prefix = '/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/gender_based_data/female/'
	age_range_ids = [0,1,2,3,4,5,6,7]

	for a in age_range_ids:
	
		male_chunk_size = int(math.ceil(male_age_group_counters_across_folds[a]))
		female_chunk_size = int(math.ceil(female_age_group_counters_across_folds[a]))

		male_overflow = []
		female_overflow = []

		male_folds_needing_adjustment = []
		female_folds_needing_adjustment = []
		
		fold_i=0
		for fold in fold_names:
			male_age_group_data = []
			female_age_group_data = []

			'''
				male_fold_i's data will be in csv/fold_i
				Similarly for female

			'''
			fold_df = pd.read_csv('/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/data/csvs/'+fold+'.csv')
			male_rec_count = 0
			female_rec_count = 0

			for index, row in fold_df.iterrows():

				#user_id, original_image, face_id, age, gender, fiducial_yaw_angle
				age = row['age']
				gender = row['gender']
				yaw_angle = row['fiducial_yaw_angle']
				
				
				if ((gender!='u') and (gender!='Nan') and (age!='None') and (gender!=' ') and (age!=' ') and (yaw_angle >= -45) and (yaw_angle <= 45)):
					age_tuple = make_tuple(age)
					age_id = get_age_range_id(age_tuple)
					
					if(age_id == a):

						d = {	
								'user_id': row['user_id'], 'original_image': row['original_image'], 
								'face_id': row['face_id'], 'original_age': age,
								'original_gender': gender, 'yaw_angle': yaw_angle,
								'original_fold_name': fold, 'age_id': age_id

							}
						if((gender == 'm')):
							if(male_rec_count < male_chunk_size):
								male_rec_count+=1
								male_age_group_data.append(d)
							else:
								male_overflow.append(d)

						elif((gender == 'f')):
							if (female_rec_count < female_chunk_size):
								female_rec_count+=1
								female_age_group_data.append(d)
							else:
								female_overflow.append(d)

			'''
			Dump the records of new fold
			'''
			if(male_rec_count != male_chunk_size):
				male_folds_needing_adjustment.append({'fold_name':fold_i,'adjustment_count_needed':(abs(male_chunk_size-male_rec_count))})

			save_path='/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/new_distributed_data_subject_exclusive/male/'
			with open(save_path+'male_fold_'+str(fold_i) + '.pkl', 'a') as f:
				pickle.dump(male_age_group_data, f, pickle.HIGHEST_PROTOCOL)


			if(female_rec_count != female_chunk_size):
				female_folds_needing_adjustment.append({'fold_name':fold_i,'adjustment_count_needed':(abs(female_chunk_size-female_rec_count))})

			save_path='/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/new_distributed_data_subject_exclusive/female/'
			with open(save_path+'female_fold_'+str(fold_i) + '.pkl', 'a') as f:
				pickle.dump(female_age_group_data, f, pickle.HIGHEST_PROTOCOL)

			print len(male_overflow)
			print len(female_overflow)
			fold_i+=1			
		
		'''
		Before exiting the age_group loop
		redistribute overflow into existing

		start = 0
		for adj in male_folds_needing_adjustment:
			remaining_data = male_overflow[start:start+adj['adjustment_count_needed']]
			print type(remaining_data)
			save_path='/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/new_distributed_data_subject_exclusive/male/'
			with open(save_path+'male_fold_'+str(adj['fold_name']) + '.pkl', 'a') as f:
				pickle.dump(remaining_data, f, pickle.HIGHEST_PROTOCOL)
			start = adj['adjustment_count_needed']+1


		start = 0
		for adj in female_folds_needing_adjustment:
			remaining_data = female_overflow[start:start+adj['adjustment_count_needed']]
			save_path='/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/new_distributed_data_subject_exclusive/female/'
			with open(save_path+'female_fold_'+str(adj['fold_name']) + '.pkl', 'a') as f:
				pickle.dump(remaining_data, f, pickle.HIGHEST_PROTOCOL)
			start = adj['adjustment_count_needed']+1
		
		'''
	
		print a

def read_new_folds():
	fold_names = ['0','1','2','3']	
	
	male_pickle_file_path_prefix = '/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/new_distributed_data_subject_exclusive/male/'
	female_pickle_file_path_prefix = '/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/new_distributed_data_subject_exclusive/female/'

	age_range_ids = [0,1,2,3,4,5,6,7]
	male_age_group_counters_across_folds = np.zeros(8)
	female_age_group_counters_across_folds = np.zeros(8)


	for fold in fold_names:
		train_ages = []

		print('Trying to read training fold: %s......' % fold)
		male_data = load_pickles(male_pickle_file_path_prefix+'male_fold_'+fold)
		male_df = pd.DataFrame(male_data)

		male_ages = np.array(male_df['age_id'])
		male_age_counter = Counter(male_ages)
		male_age_counter_dict = dict(male_age_counter)
		male_age_counter_dict = sorted(male_age_counter_dict.items(), key=lambda i: i[0])
		print male_age_counter_dict
		male_age_counter_dict = dict(male_age_counter_dict)
		for i in range(8):
			if i in male_age_counter_dict:
				male_age_group_counters_across_folds[i] += male_age_counter_dict[i]

		female_data = load_pickles(female_pickle_file_path_prefix+'female_fold_'+fold)
		female_df = pd.DataFrame(female_data)

		female_ages = np.array(female_df['age_id'])
		female_age_counter = Counter(female_ages)
		female_age_counter_dict = dict(female_age_counter)
		female_age_counter_dict = sorted(female_age_counter_dict.items(), key=lambda i: i[0])
		print female_age_counter_dict
		female_age_counter_dict = dict(female_age_counter_dict)
		for i in range(8):
			if i in female_age_counter_dict:
				female_age_group_counters_across_folds[i] += female_age_counter_dict[i]
		print(' ')
		

def main():
	male_age_group_counters_across_folds, female_age_group_counters_across_folds = read_fold_for_each_group()
	distribute_data(male_age_group_counters_across_folds,female_age_group_counters_across_folds)

	read_new_folds()


if __name__=='__main__':
	main()