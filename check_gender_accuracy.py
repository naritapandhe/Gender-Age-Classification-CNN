import pickle
import json
from pprint import pprint
from sklearn.metrics import confusion_matrix

'''
1. Read the original pickle file: fold_4_data
'''


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
	predicted_file_path = '/Users/admin/Documents/pythonworkspace/data-science-practicum/final-project/gender-age-classification/outputs/female_age_model/female_age_predictions.txt'

	#test_file_contents = load_gt_file(gtfile_path,gtfile_name)
	predictions = load_predictions(predicted_file_path)
	match_count = 0
	unmatch_count = 0
	one_off_count = 0
	data = []
	json_file = '/Volumes/Mac-B/faces-recognition/jsons/testing_dataf.json'
	
	with open(json_file) as data_file:    
		data = json.load(data_file)

	y_true = []
	y_pred = []	
	#pprint(data)

	#dict = {'fold_name': fold, 'images': inputimages, 'genders': genders,'ages':ages, 'folder_names': folder_names, 'image_names':image_names,'face_ids':face_ids}
	#fold_name = test_file_contents['fold_name']
	#if fold_name == gtfile_name:
	#	inputimages = test_file_contents['images']
	#	#ground_truth = test_file_contents['genders']
	#	ground_truth = test_file_contents['ages']

	for i in range(len(data)):
		y_true.append(data[i]['age'])
		y_pred.append(predictions[i])

		if (predictions[i] == data[i]['age']):
				match_count+=1
		if	(abs(predictions[i] - data[i]['age'])==1):
				one_off_count+=1
		#else:
		#	unmatch_count+=1	
	
	print confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4,5,6,7,	8])
	print("Matched Count: %f" % match_count)				
	print("One off Count: %f" % one_off_count)				
	#print("Unmatched Count: %f" % unmatch_count)
	accuracy = (float(match_count)/len(data))
	print("Accuracy: %f" % accuracy)
	print("One Off Accuracy: %f" % (float(one_off_count)/len(data)))
	

if __name__ == '__main__':
	main()

