import pandas as pd
import numpy as np
import PIL
from PIL import Image
import pickle
import json
from ast import literal_eval as make_tuple
from operator import itemgetter
import sys

'''
1. Read the json with predicted+actuale genders
2. Read the original image
3. Create a numpy array

'''
def load_gt_file(file_name):
	with open(file_name, 'rb') as f:
		data = json.load(f)

	return data	


def save_obj(obj,name, save_path):
	with open(save_path+name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_predictions(file_name):
	output = []
	with open(file_name) as f:
		output = f.readlines()

	output = [int(line.rstrip()) for line in output]
	return output   


width = 256
height = 256
def main():
	#1. Read the original test data json
	json_file = '/Volumes/Mac-B/faces-recognition/new_model_dicts/alldata/clubbed_testing_data.json'
	gt_data = load_gt_file(json_file)
	print len(gt_data)

	maleinputimages = []
	femaleinputimages = []

	for i in range(len(gt_data)):
		folder_name = str(gt_data[i]['folder_name'])
		face_id = str(gt_data[i]['face_id'])
		image_name = str(gt_data[i]['image_name'])
		actual_gender = str(gt_data[i]['actual_gender'])
		predicted_gender = str(gt_data[i]['predicted_gender'])
		
		image_path = '/Volumes/Mac-B/faces-recognition/aligned/'+folder_name+'/landmark_aligned_face.'+(face_id)+'.'+image_name
		image = Image.open(image_path)
			
		#Resize image
		image = image.resize((width, height), PIL.Image.ANTIALIAS)
		image_arr = np.array(image)
		
		if predicted_gender == 'm':
			maleinputimages.append(image_arr)
		else:
			femaleinputimages.append(image_arr)	
								
	print len(maleinputimages)            
	print len(femaleinputimages)
	
	dict = {'gender_name': 'm', 'images': maleinputimages}
	save_obj(dict,'male_predicted_images','/Volumes/Mac-B/faces-recognition/male_predicted/')
	print ('MALE Done')

	dict = {'gender_name': 'f', 'images': femaleinputimages}
	save_obj(dict,'female_predicted_images','/Volumes/Mac-B/faces-recognition/female_predicted/')
	print ('FEMALE Done')
	


	
		
	

if __name__ == "__main__":
	main()		


