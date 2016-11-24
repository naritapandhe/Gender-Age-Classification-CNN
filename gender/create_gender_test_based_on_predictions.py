import pickle
import json
from pprint import pprint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import collections, numpy
import itertools
import numpy as np
import collections


'''
1. Read the original pickle file: fold_4_data
'''

def save_pickle(obj,name,save_path):
	with open(save_path+name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_gt_file(file_path,file_name):
	with open(file_path+file_name+ '.pkl', 'rb') as f:
		return pickle.load(f)

def one_hot(y,test_mode):
	if test_mode == 'gender':
		y_ret = np.zeros((len(y), 2))
	else:
		y_ret = np.zeros((len(y), 8))

	y_ret[np.arange(len(y)), y.astype(int)] = 1
	return y_ret


def load_predictions(file_name):
	output = []
	with open(file_name) as f:
		output = f.readlines()


	output = [int(line.rstrip()) for line in output]
	return output   

def main():
	predicted_file_path = './new_gender_prediction.txt'
	predictions = load_predictions(predicted_file_path)
	predictions = np.array(predictions)
	print collections.Counter(predictions)



	test_fold_names = ['fold_4_data']
	#pickle_file_path_prefix = '/Volumes/Mac-B/faces-recognition/gender_neutral_data/'
	pickle_file_path_prefix = '/home/ubuntu/gender_age/gender_based_train_and_testing/gender_based_data/'


	for fold in test_fold_names:
		print('Trying to read test fold: %s......' % fold)
		gt_file = load_gt_file(pickle_file_path_prefix,fold)
		
		gt_ages = []
		gt_genders = []

		
		gt_ages = np.array(gt_file['ages'])
		gt_genders = np.array(gt_file['genders'])
		gt_images = np.array(gt_file['images'])

	
		print ("GT loaded for fold: %s" % fold)
		print gt_ages.shape
		print gt_genders.shape
		print ("Images shape:")
		print gt_images.shape

		print ("Prediction shape: ")
		print predictions.shape
	
	maleimages = []
	maleactualages = []
	maleactualgenders = []
	malepredictedgenders = []


	femaleimages = []
	femaleactualages = []
	femaleactualgenders = []
	femalepredictedgenders = []

	for i in range(len(gt_genders)):
		'''
			If the predicted gender is male:
		'''
		if (predictions[i] == 0):
			maleimages.append(gt_images[i])
			maleactualages.append(gt_ages[i])
			maleactualgenders.append(gt_genders[i])
			malepredictedgenders.append(predictions[i])

		else:
			femaleimages.append(gt_images[i])
			femaleactualages.append(gt_ages[i])
			femaleactualgenders.append(gt_genders[i])
			femalepredictedgenders.append(predictions[i])


	print ('Created testing data based on predicted genders..')            
	print ('Male Images: %i, Male Actual Gender: %i, Male Predicted Gender: %i , Male Actual Ages: %i' % (len(maleimages), len(maleactualgenders), len(malepredictedgenders), len(maleactualages)))
	print ('Female Images: %i, Female Actual Gender: %i, Female Predicted Gender: %i , Female Actual Ages: %i' % (len(femaleimages), len(femaleactualgenders), len(femalepredictedgenders), len(femaleactualages)))
	print ('')


	currDict = {'test_data_for': 'male_test_data', 'images': maleimages, 'gt_genders': maleactualgenders,'pt_genders':malepredictedgenders , 'gt_ages': maleactualages}
	save_pickle(currDict,'male_test', '/home/ubuntu/gender_age/gender_based_train_and_testing/gender_based_data/cv/male/')

	currDict = {'test_data_for': 'female_test_data', 'images': femaleimages, 'gt_genders': femaleactualgenders,'pt_genders':femalepredictedgenders , 'gt_ages': femaleactualages}
	save_pickle(currDict,'female_test', '/home/ubuntu/gender_age/gender_based_train_and_testing/gender_based_data/cv/female/')

	print("Testing data created for male and females")

if __name__ == '__main__':
	main()

