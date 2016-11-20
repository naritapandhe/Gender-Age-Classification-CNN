import pickle
import json
from pprint import pprint
from sklearn.metrics import confusion_matrix
import collections, numpy
import matplotlib.pyplot as plt
import itertools
import numpy as np



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
	#gtfile_path = '/Volumes/Mac-B/faces-recognition/new_dicts/male/'
	#gtfile_name = 'fold_4_data'
	#predicted_file_path = '/Users/admin/Documents/pythonworkspace/data-science-practicum/final-project/gender-age-classification/outputs/base_model/age_predictions.txt'
	predicted_file_path = '/Volumes/Mac-B/faces-recognition/new_model_dicts/alldata/gender_prediction.txt'

	#test_file_contents = load_gt_file(gtfile_path,gtfile_name)
	predictions = load_predictions(predicted_file_path)
	predictions[predictions == 0] = 'm'
	predictions[predictions == 1] = 'f'

	match_count = 0
	unmatch_count = 0
	one_off_count = 0
	data = []
	json_file = '/Volumes/Mac-B/faces-recognition/new_model_dicts/alldata/testing_data.json'
	
	with open(json_file) as data_file:    
		data = json.load(data_file)

	y_true = []
	y_pred = [] 
	#pprint(data)

	#dict = {'fold_name': fold, 'images': inputimages, 'genders': genders,'ages':ages, 'folder_names': folder_names, 'image_names':image_names,'face_ids':face_ids}
	#fold_name = test_file_contents['fold_name']
	#if fold_name == gtfile_name:
	#   inputimages = test_file_contents['images']
	#   #ground_truth = test_file_contents['genders']
	#   ground_truth = test_file_contents['ages']

	for i in range(len(data)):
		y_true.append(data[i]['gender'])
		p = ''
		if(predictions[i] == 0):
			p = 'm'
		else:
			p = 'f'	
		
		y_pred.append(p)

		'''
		if (predictions[i] == data[i]['age']):
				match_count+=1
		if  (abs(predictions[i] - data[i]['age'])==1):
				one_off_count+=1
		'''
		if(p==data[i]['gender']):
			match_count+=1			
		else:
		   unmatch_count+=1   

	#print y_true	

	cnf_matrix = confusion_matrix(y_true, y_pred,labels=['m','f'])	 
	print cnf_matrix  
	print("Matched Count: %f" % match_count)                
	#print("One off Count: %f" % one_off_count)              
	#print("Unmatched Count: %f" % unmatch_count)
	accuracy = (float(match_count)/len(data))
	print("Accuracy: %f" % accuracy)
	print (collections.Counter(y_true))
	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=['m','f'],title='Confusion matrix, without normalization')
	plt.show()
	#print("One Off Accuracy: %f" % (float(one_off_count)/len(data)))
	

if __name__ == '__main__':
	main()

