import pickle

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
	file_path = '/Volumes/Mac-B/faces-recognition/new_dicts/aligned/'
	file_name = 'fold_4_data'
	predicted_file_path = '/Users/admin/Documents/pythonworkspace/data-science-practicum/final-project/gender-age-classification/outputs/base_model/age_predictions.txt'

	test_file_contents = load_gt_file(file_path,file_name)
	predictions = load_predictions(predicted_file_path)
	match_count = 0
	unmatch_count = 0

	#dict = {'fold_name': fold, 'images': inputimages, 'genders': genders,'ages':ages, 'folder_names': folder_names, 'image_names':image_names,'face_ids':face_ids}
	fold_name = test_file_contents['fold_name']
	if fold_name == file_name:
		inputimages = test_file_contents['images']
		#ground_truth = test_file_contents['genders']
		ground_truth = test_file_contents['ages']

		for i in range(len(inputimages)):
			if (predictions[i] == ground_truth[i]):
				match_count+=1
			else:
				unmatch_count+=1	
	
		print("Matched Count: %f" % match_count)				
		print("Unmatched Count: %f" % unmatch_count)
		accuracy = (float(match_count)/len(inputimages))
		print("Accuracy: %f" % accuracy)				

if __name__ == '__main__':
	main()

