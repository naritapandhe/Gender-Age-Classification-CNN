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
		actual_age = str(gt_data[i]['actual_age_id'])
		
		dict = {'folder_name': folder_name, 'face_id': face_id, 'image_name': image_name, 'actual_gender': actual_gender, 'predicted_gender':predicted_gender,'actual_age':actual_age}
		if predicted_gender == 'm':
			maleinputimages.append(dict)
		else:
			femaleinputimages.append(dict)	



	print len(maleinputimages)
	print len(femaleinputimages)

	with open('/Volumes/Mac-B/faces-recognition/new_model_dicts/m/male_predicted_genders.json', 'w') as fout:
		json.dump(maleinputimages, fout)	

	with open('/Volumes/Mac-B/faces-recognition/new_model_dicts/f/female_predicted_genders.json', 'w') as fout:
		json.dump(femaleinputimages, fout)	

	print "Saved jsons"
		


if __name__ == '__main__':
	main()
