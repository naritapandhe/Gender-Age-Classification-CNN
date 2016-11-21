import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
import sys
import pickle
import random


train_mode = 'age'
train_for = 'f'
model_mode = 'test'

def one_hot(y):
    y_ret = np.zeros((len(y), 8))
    y_ret[np.arange(len(y)), y.astype(int)] = 1
    return y_ret

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


#Read the data from pickles
pickle_file_paths = ['female_predicted_images']
#pickle_file_paths = ['fold_0_data','fold_1_data','fold_2_data','fold_3_data','fold_4_data']
#pickle_file_path_prefix = '/Volumes/Mac-B/faces-recognition/new_dicts/aligned/'
pickle_file_path_prefix = '/home/ubuntu/gender_age/f/'

X = []
y = []
X_test = []
folder_names = []
image_names = []
face_ids = []



for pf in pickle_file_paths:
    pfile = load_obj(pickle_file_path_prefix+pf)
    images = []
    labels = []

    if pf == 'female_predicted_images':
        images = (pfile['images'])
        
        images = np.array(images)
        X_test.append(images)

    else:

        #dict = {'fold_name': fold, 'images': inputimages, 'labels': genders}
        images = (pfile['images'])

        if train_mode == 'age':
            labels = (pfile['ages'])

            
        images = np.array(images)
        labels = np.array(labels)

        labels = one_hot(labels)
        X.append(images)
        y.append(labels)

X_test = np.array(X_test)
X_test = np.vstack(X_test)

'''

X = np.array(X)
X = np.vstack(X)

y = np.array(y)
y = np.vstack(y)


print "Before starting..."
print X.shape
print y.shape
X_train = X
y_train = y

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
print "After read all, train shapes: "
print X_train.shape
print y_train.shape

print ("Validation shape: ")
print X_val.shape
print y_val.shape
'''

print "Test shape: "
print X_test.shape


print ('Training, Validation and Test dataset created!!')

#train_size = X_train.shape[0]
image_size = 227
num_channels = 3
batch_size = 50
patch_size = 3
width = 256
height = 256
new_width = 227
new_height = 227
if train_mode == 'gender':
    num_labels = 2
else:
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
                  tf.nn.l2_loss(w1) + tf.nn.l2_loss(b1) 
                )

# Add the regularization term to the loss.
cross_entropy += 5e-4 * regularizers

prediction=tf.nn.softmax(fc3)

learning_rate = tf.placeholder(tf.float32, shape=[])
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

sess.run(init_op)

num_steps = 25000

#print "Restoring the model and predicting....."
ckpt = tf.train.get_checkpoint_state("/home/ubuntu/gender_age/f/saved_model/")
if ckpt and ckpt.model_checkpoint_path:
    # Restores from checkpoint
    saver.restore(sess, "/home/ubuntu/gender_age/f/saved_model/model.ckpt")
    print "Model loaded"
    for i in range(num_steps):
            
            #run model on test
            if (i % 1000 == 0):
                print i
                preds = []
                for j in range(0,X_test.shape[0],batch_size):
                    X_batch = X_test[j:j+batch_size,:,:,:]
              
                    #Center Crop
                    left = (width - new_width)/2
                    top = (height - new_height)/2
                    right = (width + new_width)/2
                    bottom = (height + new_height)/2
                    X_batch = X_batch[:,left:right,top:bottom,:]

                    feed_dict={tfx:X_batch}
                    p = sess.run(prediction, feed_dict=feed_dict)
                    preds.append(np.argmax(p, 1))

                pred = np.concatenate(preds)
                np.savetxt('female_age_prediction.txt',pred,fmt='%.0f') 


    print "Done predicitng..."               

else:
            print "No checkpoint file found"        
    

