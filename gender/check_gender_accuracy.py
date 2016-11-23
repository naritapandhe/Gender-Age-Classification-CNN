import pickle
import json
from pprint import pprint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import collections, numpy
import matplotlib.pyplot as plt
import itertools
import numpy as np


test_mode = 'gender'


'''
1. Read the original pickle file: fold_4_data
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

def load_gt_file(file_path,file_name):
	with open(file_path+file_name+ '.pkl', 'rb') as f:
		return pickle.load(f)

def one_hot(y):
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
	'''
	for i in range(len(output)):
		if(output[i]==0):
			output[i]='m'
		else:
			output[i]='f'   
	'''     
	return output   

def main():
	predicted_file_path = '/Volumes/Mac-B/faces-recognition/gender/gender_prediction.txt'
	#test_file_contents = load_gt_file(gtfile_path,gtfile_name)
	predictions = load_predictions(predicted_file_path)
	

	test_fold_names = ['fold_4_data']
	#pickle_file_path_prefix = '/Volumes/Mac-B/faces-recognition/gender_neutral_data/'
	pickle_file_path_prefix = '/home/ubuntu/gender_age/gender_neutral_data/'


	for fold in test_fold_names:
		print('Trying to read test fold: %s......' % fold)
		gt_file = load_gt_file(pickle_file_path_prefix+fold)
		
		gt_ages = []
		gt_genders = []

		'''
		Load all the ground truth ages
		'''
		for i in range(len(gt_file)):
			current_file = gt_file[i]
			ages = np.array(current_file['ages'])
			genders = np.array(current_file['genders'])
			one_hot1 = one_hot(ages)
			gt_ages.append(one_hot1)
			gt_genders.append(genders)


		gt_ages = np.array(gt_ages)
		gt_ages = np.vstack(gt_ages)

		gt_genders = np.array(gt_genders)
		gt_genders = np.vstack(gt_genders)
		
		print ("GT loaded for fold: %s" % fold)
		print gt_ages.shape
		print gt_genders.shape
	
	y_true = []
	y_pred = []

	for i in range(len(gt_ages)):
		y_true.append(gt_ages[i])
		y_pred.append(predictions[i])

	print("#GT: %i, PT: %i" %(len(y_true), len(y_pred)))	


	male_count = 0			
	male_exact_match = 0
	male_one_off_count = 0

	female_count = 0
	female_exact_match = 0
	female_one_off_count = 0

	one_off_count = 0


	for i in range(len(y_true)):
		if  (abs(y_true[i] - y_pred[i])<=1):
				one_off_count+=1

		if(gt_genders[i]==0):
			#Male
			male_count += 1
			if(y_true[i] == y_pred[i]):
				male_exact_match+=1
				
			if  (abs(y_true[i] - y_pred[i])<=1):
				male_one_off_count+=1	

		else:
			#Female
			female_count += 1
			if(y_true[i] == y_pred[i]):
				female_exact_match+=1
				
			if  (abs(y_true[i] - y_pred[i])<=1):
				female_one_off_count+=1	

			

	print('Exact: %f, One-Off: %f' % (accuracy_score(y_true, y_pred),(float(one_off_count/len(y_true)))*100))	
	print('#Males: %i, Exact: %f, One-Off: %f' % (male_count,((float(male_exact_match/male_count))*100),((float(male_one_off_count/male_count))*100)))	
	print('#Females: %i, Exact: %f, One-Off:: %f' % (female_count,((float(female_exact_match/female_count))*100),((float(female_one_off_count/female_count))*100)))		

	# Plot non-normalized confusion matrix
	cnf_matrix = confusion_matrix(y_true, y_pred,labels=[0,1])	 
	print cnf_matrix  
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=['m','f'],title='Confusion matrix, without normalization')
	plt.show()
	

if __name__ == '__main__':
	main()

