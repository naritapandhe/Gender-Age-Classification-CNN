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
	image_path = '/Volumes/Mac-B/faces-recognition/gillevildata/alldata/'
	with open(image_path+name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)


def create_data(fold_names):
	print ('Creating data......')
	i = 0;
	width = 256
	height = 256
	
	for fold in fold_names:
		train_subset_path = '/Volumes/Mac-B/AgeGenderDeepLearning-master/Folds/train_val_txt_files_per_fold/'+fold+'/gender_train_subset.txt'
		val_path = '/Volumes/Mac-B/AgeGenderDeepLearning-master/Folds/train_val_txt_files_per_fold/'+fold+'/gender_val.txt'
		test_path = '/Volumes/Mac-B/AgeGenderDeepLearning-master/Folds/train_val_txt_files_per_fold/'+fold+'/gender_test.txt'
		
		train_df = pd.read_csv(train_subset_path, sep=" ", header = None)
		val_df = pd.read_csv(val_path, sep=" ", header = None)
		test_df = pd.read_csv(test_path, sep=" ", header = None)

		print ("Length of records in train: %s = %i" % (fold, len(train_df)))
		print ("Length of records in val: %s = %i" % (fold, len(val_df)))
		print ("Length of records in test: %s = %i" % (fold, len(test_df)))

		images_names = []
		images = []
		genders = []
		for index, row in train_df.iterrows():
			train_image = '/Volumes/Mac-B/faces-recognition/aligned/'+row[0]
			image = Image.open(train_image)
	
			#Resize image
			image = image.resize((width, height), PIL.Image.ANTIALIAS)
			image_arr = np.array(image)
			
			images_names.append(row[0])
			images.append(image_arr)
			genders.append(row[1])

		dict = {'fold_name':fold, 'image_name': images_names, 'images':images, 'genders': genders, 'dataset_type': 'train'}
		print (fold)            
		print ('Train: '+str(len(images_names))+":"+str(len(images))+":"+str(len(genders)))
		save_obj(dict,fold+'_train')
		
		val_images_names = []
		val_images = []
		val_genders = []
		for index, row in val_df.iterrows():
			val_image = '/Volumes/Mac-B/faces-recognition/aligned/'+row[0]
			image = Image.open(val_image)
	
			#Resize image
			image = image.resize((width, height), PIL.Image.ANTIALIAS)
			image_arr = np.array(image)
			
			val_images_names.append(row[0])
			val_images.append(image_arr)
			val_genders.append(row[1])

		dict = {'fold_name':fold, 'image_name': val_images_names, 'images':val_images, 'genders': val_genders, 'dataset_type': 'val'}
		print ('Val:' + str(len(val_images_names))+":"+str(len(val_images))+":"+str(len(val_genders)))
		save_obj(dict,fold+'_val')
		
		test_images_names = []
		test_images = []
		test_genders = []
		for index, row in test_df.iterrows():
			val_image = '/Volumes/Mac-B/faces-recognition/aligned/'+row[0]
			image = Image.open(val_image)
	
			#Resize image
			image = image.resize((width, height), PIL.Image.ANTIALIAS)
			image_arr = np.array(image)

			test_images_names.append(row[0])
			test_images.append(image_arr)
			test_genders.append(row[1])

		dict = {'fold_name':fold, 'image_name': test_images_names, 'images':test_images, 'genders': test_genders, 'dataset_type': 'val'}
		print ('Test: '+str(len(test_images_names))+":"+str(len(test_images))+":"+str(len(test_genders)))
		print ('')

		save_obj(dict,fold+'_test')
		print('Done: {%s}/{%s} fold' % (fold, fold_names))


	print "Dataset Created"	


		
def main():
	fold_names=['test_fold_is_0','test_fold_is_1','test_fold_is_2','test_fold_is_3','test_fold_is_4']
	create_data(fold_names)

if __name__ == "__main__":
	main()


