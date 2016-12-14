'''
1. Read every subject exclusive fold
2. For every data point, read the image
3. Store the dict with images
'''
import pandas as pd
import numpy as np
import PIL
from PIL import Image
import pickle
import random
import json
from ast import literal_eval as make_tuple
from operator import itemgetter
from pprint import pprint
from skimage import exposure
import itertools

def save_pickle(obj,name,save_path):
	with open(save_path+name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(path,name):
	with open(path+name + '.pkl', 'rb') as f:
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

def read_fold_and_images(fold_names, pickle_path, fold_image_prefix):
	width = 256
	height = 256
	i=0
	
	for fold in fold_names:
		maleinputimages = []
		malegenders = []
		maleages = []

		femaleinputimages = []
		femalegenders = []
		femaleages = []

		print("Reading fold: %s" % fold)
		
		train_ages = []
	
		print('Trying to read training fold: %s......' % fold)
		
		fold_file = load_pickles(pickle_path+fold)
		merged = list(itertools.chain(*fold_file))
		df = pd.DataFrame(merged)
		
		for index, row in df.iterrows():
			yaw_angle = row['yaw_angle']
			gender = row['original_gender']
			age = row['original_age']
					
			if ((gender!='u') and (gender!='Nan') and (age!='None') and (gender!=' ') and (age!=' ') and (yaw_angle >= -45) and (yaw_angle <= 45)):
					folder_name = row['user_id']
					image_name = row['original_image']
					face_id = row['face_id']
					
					age_id = row['age_id']
					
					image_path = fold_image_prefix+folder_name+'/landmark_aligned_face.'+str(face_id)+'.'+image_name
					image = Image.open(image_path)
			
					#Resize image
					image = image.resize((width, height), PIL.Image.ANTIALIAS)
					image_arr = np.array(image)
					
					

					if(gender == 'm'):
						g=0
						maleinputimages.append(image_arr)
						malegenders.append(g)
						maleages.append(age_id)
					else:
						g=1
						femaleinputimages.append(image_arr)
						femalegenders.append(g)
						femaleages.append(age_id)
		
		print('Done: {0}/{1} folds'.format(i, len(fold_names)))

		print ('Fold Name: %s' % fold)            
		#print ('Male Images: %i, Male Gender: %i, Male Ages: %i' % (len(maleinputimages), len(malegenders), len(maleages)))
		print ('Female Images: %i, Female Gender: %i, Female Ages: %i' % (len(femaleinputimages), len(femalegenders), len(femaleages)))            
		print ('')



		#currDict = {'fold_name': fold, 'images': maleinputimages, 'genders': malegenders, 'ages': maleages}
		#save_pickle(currDict, fold, '/media/narita/My Passport/Gender-Age Classification/subject_exclusive_distributed_data/male/')

		currDict = {'fold_name': fold, 'images': femaleinputimages, 'genders': femalegenders, 'ages': femaleages}
		save_pickle(currDict,'female_'+fold, '/media/narita/My Passport/Gender-Age Classification/subject_exclusive_distributed_data/female/')
		
def create_crossval_data(fold_names):

	total_files = len(fold_names)
	genders = ['female']
	cv_train = []
	cv_val=[]

	for gender in genders:
		cv_train = []
		cv_val=[]

		#prefix_fold_name=gender+'_'
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
		print (' ')	
		
		for i in range(len(cv_train)):
			train_files = cv_train[i]
			train_data = []
			val_data = []

			for ii in range(len(train_files)):
				train_file_data = load_pickle('/media/narita/My Passport/Gender-Age Classification/subject_exclusive_distributed_data/'+gender+'/',train_files[ii])
				train_data.append(train_file_data)

			val_data = load_pickle('/media/narita/My Passport/Gender-Age Classification/subject_exclusive_distributed_data/'+gender+'/',cv_val[i])	
			print (cv_val[i])
			print len(val_data)

			save_pickle(train_data,gender+'_cv_train_'+str(i), '/media/narita/My Passport/Gender-Age Classification/subject_exclusive_distributed_data/female/')
			save_pickle(val_data,gender+'_cv_val_'+str(i), '/media/narita/My Passport/Gender-Age Classification/subject_exclusive_distributed_data/female/')

			print("Train and val created for fold: %i" % i)		
		
		


def main():
	female_subject_exclusive_folds = ['female_fold_0_data','female_fold_1_data','female_fold_2_data','female_fold_3_data']
	female_subject_exclusive_fold_prefix = '/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/new_distributed_data_subject_exclusive/female/'
	#read_fold_and_images(female_subject_exclusive_folds,female_subject_exclusive_fold_prefix,'/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/data/aligned/')
	create_crossval_data(female_subject_exclusive_folds)
	
if __name__=='__main__':
	main()