import pandas as pd
import numpy as np
import PIL
from PIL import Image
import pickle
import json
from ast import literal_eval as make_tuple
from operator import itemgetter
import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


'''
1. Read the json with predicted+actuale genders
2. Read the original image
3. Create a numpy array

'''
def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

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
	json_file = '/Volumes/Mac-B/faces-recognition/new_model_dicts/f/female_predicted_genders.json'
	#json_file = '/Volumes/Mac-B/faces-recognition/new_model_dicts/m/male_predicted_genders.json'
	gt_data = load_gt_file(json_file)
	print len(gt_data)

	output = []
	with open('/Users/admin/Documents/pythonworkspace/data-science-practicum/final-project/gender-age-classification/age/female_age_prediction.txt') as f:
	#with open('/Users/admin/Documents/pythonworkspace/data-science-practicum/final-project/gender-age-classification/age/male_prediction.txt') as f:
		output = f.readlines()


	output = [int(line.rstrip()) for line in output]
	print len(output)

	maleinputimages = []
	match_count = 0
	one_off_count = 0
	y_true = []
	y_pred = []
	for i in range(len(gt_data)):
		folder_name = str(gt_data[i]['folder_name'])
		face_id = str(gt_data[i]['face_id'])
		image_name = str(gt_data[i]['image_name'])
		actual_gender = str(gt_data[i]['actual_gender'])
		predicted_gender = str(gt_data[i]['predicted_gender'])
		actual_age = int(gt_data[i]['actual_age'])
		y_true.append(actual_age)
		y_pred.append(output[i])
		
		dict = {
					'folder_name': folder_name, 
					'face_id': face_id, 
					'image_name': image_name, 
					'actual_gender': actual_gender, 
					'predicted_gender':predicted_gender,
					'actual_age_id':actual_age,
					'predicted_age_id':output[i]
				}

		maleinputimages.append(dict)
		if(output[i]==actual_age):
			match_count+=1
		
		if((abs(output[i] - actual_age)) <= 1):
			one_off_count+=1
			

	print len(maleinputimages)
	
	with open('/Volumes/Mac-B/faces-recognition/new_model_dicts/f/female_predictednactual_agengender.json', 'w') as fout:
		json.dump(maleinputimages, fout)	
	#with open('/Volumes/Mac-B/faces-recognition/new_model_dicts/m/male_predictednactual_agengender.json', 'w') as fout:
	#	json.dump(maleinputimages, fout)	

	
	print "Saved jsons"
	print match_count
	print one_off_count

	# Plot non-normalized confusion matrix
	classes = [0,1,2,3,4,5,6,7]
	cnf_matrix = confusion_matrix(y_true, y_pred,labels=classes)	 
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=classes,title='Confusion matrix, without normalization',normalize=True)
	plt.show()

	


if __name__ == '__main__':
	main()
