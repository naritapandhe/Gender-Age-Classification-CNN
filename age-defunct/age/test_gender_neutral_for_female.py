import tensorflow as tf
import numpy as np
import sys
import pickle
import random
from sklearn.utils import shuffle

train_mode = 'age'

def load_train_file(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)

def load_val_file(name):
	with open(name+ '.pkl', 'rb') as f:
		return pickle.load(f)

def one_hot(y):
	if train_mode == 'gender':
		y_ret = np.zeros((len(y), 2))
	else:
		y_ret = np.zeros((len(y), 8))

	y_ret[np.arange(len(y)), y.astype(int)] = 1
	return y_ret

def load_test_file(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)



def train_and_test():

		#List of cv folds
		cv_fold_names = ['0','1','2','3']
		#pickle_file_path_prefix = '/Volumes/Mac-B/faces-recognition/gender_neutral_data/'
		train_pickle_file_path_prefix = '/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/gender_neutral_data/'
		past_tacc = 0
		past_tloss = 3.0


		test_fold_names = ['predicted_females_test']
		pickle_file_path_prefix = '/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/gender_based_data/final_test_data_based_on_predicted_genders/female/'


		print('Trying to read test fold: %s......' % test_fold_names[0])
		test_file = load_test_file(pickle_file_path_prefix+test_fold_names[0])
			
			
		test_images = []
		test_ages = []
		imgs = np.array(test_file['images'])
		ages = np.array(test_file['gt_ages'])
		one_hot1 = one_hot(ages)
		test_images.append(imgs)
		test_ages.append(one_hot1)	

		test_images = np.array(test_images)
		test_images = np.vstack(test_images)
		
		test_ages = np.array(test_ages)
		test_ages = np.vstack(test_ages)
		
		X_test = test_images
		y_test = test_ages

		print ("TAdadad")
		print X_test.shape

	

		#print ('Training, Validation done for fold: %s\n' % fold)
		image_size = 227
		num_channels = 3
		batch_size = 50
		width = 256
		height = 256
		new_width = 227
		new_height = 227
		num_labels = 8


		sess = tf.InteractiveSession()

		def data_type():
		  """Return the type of the activations, weights, and placeholder variables."""
		  return tf.float32

		def weight_variable(shape):
			initial = tf.truncated_normal(shape, stddev=0.01)
			return initial

		def bias_variable(shape):
			initial = tf.constant(0.1, shape=shape)
			return initial

		def conv2d(x, W, stride=[1,1,1,1], pad='SAME'):
			return tf.nn.conv2d(x, W, strides=stride, padding=pad)

		def max_pool(x,k,stride=[1,1,1,1],pad='SAME'):
			return tf.nn.max_pool(x, k, strides=stride,padding=pad)

		def accuracy(predictions, labels):
			return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
					/ predictions.shape[0])


		tfx = tf.placeholder(tf.float32, shape=[None,image_size,image_size,num_channels])
		tfy = tf.placeholder(tf.float32, shape=[None,num_labels])

		#Conv Layer 1
		w1 = tf.Variable(weight_variable([7,7,3,96]),name="w1")    
		b1 = tf.Variable(bias_variable([96]),name="b1")
		c1 = tf.nn.relu(conv2d(tfx,w1,stride=[1,4,4,1],pad='SAME') + b1)
		mxp1 = max_pool(c1,k=[1,3,3,1],stride=[1,2,2,1],pad='VALID')
		lrn1 = tf.nn.local_response_normalization(mxp1, alpha=0.0001, beta=0.75)

		#Conv Layer 2
		w2 = tf.Variable(weight_variable([5,5,96,256]),name="w2")    
		b2 = tf.Variable(bias_variable([256]),name="b2")
		c2 = tf.nn.relu(conv2d(lrn1,w2,stride=[1,1,1,1],pad='SAME') + b2)
		mxp2 = max_pool(c2,k=[1,3,3,1],stride=[1,2,2,1],pad='SAME')
		lrn2 = tf.nn.local_response_normalization(mxp2, alpha=0.0001, beta=0.75)

		#Conv Layer 3
		w3 = tf.Variable(weight_variable([3,3,256,384]),name="w3")    
		b3 = tf.Variable(bias_variable([384]),name="b3")
		c3 = tf.nn.relu(conv2d(lrn2,w3,stride=[1,1,1,1],pad='SAME') + b3)
		mxp3 = max_pool(c3,k=[1,3,3,1],stride=[1,2,2,1],pad='SAME')

		#FC Layer 1
		wfc1 = tf.Variable(weight_variable([7 * 7 * 384, 512]),name="wfc1")    
		bfc1 = tf.Variable(bias_variable([512]),name="bfc1")
		mxp1_flat = tf.reshape(mxp3, [-1, 7 * 7 * 384])
		fc1 = tf.nn.relu(tf.matmul(mxp1_flat, wfc1) + bfc1)
		dfc1 = tf.nn.dropout(fc1, 0.5)

		#FC Layer 2
		wfc2 = tf.Variable(weight_variable([512, 512]),name="wfc2")    
		bfc2 = tf.Variable(bias_variable([512]),name="bfc2")
		fc2 = tf.nn.relu(tf.matmul(dfc1, wfc2) + bfc2)
		dfc2 = tf.nn.dropout(fc2, 0.7)


		#FC Layer 3
		wfc3 = tf.Variable(weight_variable([512, num_labels]),name="wfc3")  
		bfc3 = tf.Variable(bias_variable([num_labels]),name="bfc3")
		fc3 = (tf.matmul(dfc2, wfc3) + bfc3)

		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc3,tfy))
		# L2 regularization for the fully connected parameters.
		regularizers = (  tf.nn.l2_loss(wfc3) + tf.nn.l2_loss(bfc3) +
					  tf.nn.l2_loss(wfc2) + tf.nn.l2_loss(bfc2) +
					  tf.nn.l2_loss(wfc1) + tf.nn.l2_loss(bfc1) +
					  tf.nn.l2_loss(w2) + tf.nn.l2_loss(b2) +
					  tf.nn.l2_loss(w1) + tf.nn.l2_loss(b1) +
					  tf.nn.l2_loss(w3) + tf.nn.l2_loss(b3)
					)
	
		# Add the regularization term to the loss.
		cross_entropy += 5e-4 * regularizers

		prediction=tf.nn.softmax(fc3)

		learning_rate = tf.placeholder(tf.float32, shape=[])
		
		#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
		#train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
		# Optimizer: set up a variable that's incremented once per batch and
		# controls the learning rate decay.
		batch = tf.Variable(0)

		learning_rate = tf.train.exponential_decay(
		  1e-3,                # Base learning rate.
		  batch * batch_size,  # Current index into the dataset.
		  10000,          		# Decay step.
		  0.0005,                # Decay rate.
		  staircase=True)
		
		# Use simple momentum for the optimization.
		train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cross_entropy,global_step=batch)


		# Add an op to initialize the variables.
		init_op = tf.initialize_all_variables()

		# Add ops to save and restore all the variables.
		saver = tf.train.Saver()

		sess.run(init_op)

		num_steps = 100
		pathhh = '/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/gender_neutral_data/final_data/saved_model/' 

		ckpt = tf.train.get_checkpoint_state(pathhh)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(sess, pathhh+"model.ckpt")
			print "Model loaded"
			for i in range(num_steps):
				if (i % 10 == 0):
					test_accuracies = []
					test_losses = []
					preds = []
					for j in range(0,X_test.shape[0],batch_size):
						X_batch = X_test[j:j+batch_size,:,:,:]
						y_batch = y_test[j:j+batch_size,:]
				  
						#Center Crop
						left = (width - new_width)/2
						top = (height - new_height)/2
						right = (width + new_width)/2
						bottom = (height + new_height)/2
						X_batch = X_batch[:,left:right,top:bottom,:]

						feed_dict={tfx:X_batch,tfy:y_batch}
						l, predictions = sess.run([cross_entropy,prediction], feed_dict=feed_dict)   
						test_accuracies.append(accuracy(predictions,y_batch))
						test_losses.append(l)
						preds.append(np.argmax(predictions, 1))


					pred = np.concatenate(preds)
					np.savetxt('/home/narita/Documents/pythonworkspace/data-science-practicum/gender-age-classification/gender_neutral_data/final_data/gender_neutral_age_prediction_for_predicted_females.txt',pred,fmt='%.0f') 
					print('Predictions saved...')
		else:
			print("Model not found")
					


def main():
	train_and_test()

if __name__=='__main__':
	main()	
	

