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
1. Read the predicted output
2. Read the original test data
3. Add a new key:value in existing json. key: predicted_gender

'''
def load_gt_file(file_name):
	with open(file_name, 'rb') as f:
		data = json.load(f)

	return data	


def load_predictions(file_name):
	output = []
	with open(file_name) as f:
		output = f.readlines()

	output = [int(line.rstrip()) for line in output]
	return output   


def main():
	#1. Read the predictions
	predicted_file_path = '/Volumes/Mac-B/faces-recognition/new_model_dicts/alldata/gender_prediction.txt'
	predictions = load_predictions(predicted_file_path)
	predictions[predictions == 0] = 'm'
	predictions[predictions == 1] = 'f'

	#2. Read the original test data json
	json_file = '/Volumes/Mac-B/faces-recognition/new_model_dicts/alldata/testing_data.json'
	gt_data = load_gt_file(json_file)

	clubbed_data = []
	for i in range(len(gt_data)):
		if(predictions[i] == 0):
			p = 'm'
		else:
			p = 'f'	
		
		dict = {
					'folder_name': str(gt_data[i]['folder_name']), 
					'image_name': str(gt_data[i]['image_name']), 
					'face_id': str(gt_data[i]['face_id']), 
					'actual_gender': str(gt_data[i]['gender']), 
					'predicted_gender': str(p),
					'actual_age_id':str(gt_data[i]['age'])
				}
		clubbed_data.append(dict)		


	print('Dumping the clcubbed data....')		
	with open('/Volumes/Mac-B/faces-recognition/new_model_dicts/alldata/clubbed_testing_data.json', 'w') as fout:
		json.dump(clubbed_data, fout)	

		
	

if __name__ == "__main__":
	main()		


