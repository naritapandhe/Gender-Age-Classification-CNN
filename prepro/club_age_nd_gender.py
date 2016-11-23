'''
1. Read data from age_train_subset from fold_X
3. Check if there are any duplicates
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


def load_age_train_file(prefix,name):
    with open(prefix+name+'/'+ name + '_train.pkl', 'rb') as f:
        return pickle.load(f)

def load_age_val_file(prefix,name):
    with open(prefix+name +'/'+ name+ '_val.pkl', 'rb') as f:
        return pickle.load(f)


#Read the data from pickles
fold_names = ['test_fold_is_0','test_fold_is_1','test_fold_is_2','test_fold_is_3','test_fold_is_4']
#pickle_file_path_prefix = '/Volumes/Mac-B/faces-recognition/gillevildata/alldata/'
pickle_file_path_prefix = '/home/ubuntu/gender_age/alldata/'

def create_data(fold_names):
	print ('Creating data......')
	i = 0;
	width = 256
	height = 256
	
	for fold in fold_names:
		age_train_subset_path = '/Volumes/Mac-B/AgeGenderDeepLearning-master/Folds/train_val_txt_files_per_fold/'+fold+'/age_train_subset.txt'
		age_val_path = '/Volumes/Mac-B/AgeGenderDeepLearning-master/Folds/train_val_txt_files_per_fold/'+fold+'/age_val.txt'
		age_test_path = '/Volumes/Mac-B/AgeGenderDeepLearning-master/Folds/train_val_txt_files_per_fold/'+fold+'/age_test.txt'


		gender_train_subset_path = '/Volumes/Mac-B/AgeGenderDeepLearning-master/Folds/train_val_txt_files_per_fold/'+fold+'/gender_train_subset.txt'
		val_path = '/Volumes/Mac-B/AgeGenderDeepLearning-master/Folds/train_val_txt_files_per_fold/'+fold+'/gender_val.txt'
		test_path = '/Volumes/Mac-B/AgeGenderDeepLearning-master/Folds/train_val_txt_files_per_fold/'+fold+'/gender_test.txt'

		age_train_df = pd.read_csv(age_train_subset_path, sep=" ", header = None)
		age_val_df = pd.read_csv(age_val_path, sep=" ", header = None)
		age_test_df = pd.read_csv(age_test_path, sep=" ", header = None)

		gender_train_df = pd.read_csv(gender_train_subset_path, sep=" ", header = None)
		gender_val_df = pd.read_csv(val_path, sep=" ", header = None)
		gender_test_df = pd.read_csv(test_path, sep=" ", header = None)

		s1 = pd.merge(age_train_df, gender_train_df, how='inner', on=[0])
		s2 = pd.merge(age_val_df,gender_val_df,on=[0])
		s3 = pd.merge(age_test_df,gender_test_df,on=[0])

		
		print ("Length of records in age train: %s = %i" % (fold, len(age_train_df)))
		print len(s1)		

		print ("Length of records in age val: %s = %i" % (fold, len(age_val_df)))
		print len(s2)	

		print ("Length of records in age val: %s = %i" % (fold, len(age_test_df)))
		print len(s3)

		print ("\n")

def main():
	fold_names=['test_fold_is_0','test_fold_is_1','test_fold_is_2','test_fold_is_3','test_fold_is_4']
	create_data(fold_names)

if __name__ == "__main__":
	main()

	