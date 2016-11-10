import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
import sys
import pickle
np.set_printoptions(threshold=np.nan)


def one_hot(y):
    y_ret = np.zeros((len(y), 2))
    y_ret[np.arange(len(y)), y.astype(int)] = 1
    return y_ret

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


#Read the data from pickles
pickle_file_paths = ['fold_0_data']
pickle_file_path_prefix = '/Users/admin/Documents/pythonworkspace/data-science-practicum/final-project/gender-age-classification/data/'

X = []
y = []
for pf in pickle_file_paths:
	pfile = load_obj(pickle_file_path_prefix+pf)

	#dict = {'fold_name': fold, 'images': inputimages, 'labels': genders}
	X = pfile['images']
	y = pfile['labels']
	

X = np.array(X)
y = np.array(y)

indices = np.where(y =='nan')
X = np.delete(X,indices,axis=0)
y = np.delete(y, indices)

indices = np.where(y =='u')
X = np.delete(X,indices,axis=0)
y = np.delete(y, indices)

y[y == 'm'] = 0
y[y == 'f'] = 1

X = X.astype(np.float)
y = y.astype(np.float)

print X.shape
print y.shape

y = one_hot(y)
print y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


image_size = 227
num_channels = 3
num_labels=2
batch_size = 64
patch_size = 3

sess = tf.InteractiveSession()


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
w1 = tf.Variable(weight_variable([7,7,3,96]))    
b1 = tf.Variable(bias_variable([96]))
c1 = tf.nn.relu(conv2d(tfx,w1,stride=[1,3,3,1]) + b1)
mxp1 = max_pool(c1,k=[1,3,3,1],stride=[1,2,2,1])
lrn1 = tf.nn.local_response_normalization(mxp1, alpha=0.0001, beta=0.75)

#Conv Layer 2
w2 = tf.Variable(weight_variable([5,5,96,256]))    
b2 = tf.Variable(bias_variable([256]))
c2 = tf.nn.relu(conv2d(lrn1,w2,stride=[1,1,1,1],pad='VALID') + b2)
mxp2 = max_pool(c2,k=[1,3,3,1],stride=[1,2,2,1])
lrn2 = tf.nn.local_response_normalization(mxp2, alpha=0.0001, beta=0.75)

#Conv Layer 3
w3 = tf.Variable(weight_variable([3,3,256,384]))    
b3 = tf.Variable(bias_variable([384]))
c3 = tf.nn.relu(conv2d(lrn2,w3,stride=[1,1,1,1],pad='VALID') + b3)
mxp3 = max_pool(c3,k=[1,3,3,1],stride=[1,2,2,1])

#FC Layer 1
wfc1 = tf.Variable(weight_variable([8 * 8 * 384, 512]))    
bfc1 = tf.Variable(bias_variable([512]))
mxp1_flat = tf.reshape(mxp3, [-1, 8 * 8 * 384])
fc1 = tf.nn.relu(tf.matmul(mxp1_flat, wfc1) + bfc1)
dfc1 = tf.nn.dropout(fc1, 0.5)


#FC Layer 2
wfc2 = tf.Variable(weight_variable([512, 512]))    
bfc2 = tf.Variable(bias_variable([512]))
fc2 = tf.nn.relu(tf.matmul(dfc1, wfc2) + bfc2)
dfc2 = tf.nn.dropout(fc2, 0.7)


#FC Layer 2
wfc3 = tf.Variable(weight_variable([512, num_labels]))  
bfc3 = tf.Variable(bias_variable([num_labels]))
fc3 = (tf.matmul(dfc2, wfc3) + bfc3)
#fc3 = tf.reshape(fc3, [-1, num_labels])
#print "fc3.get_shape"
#print fc3.get_shape


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc3,tfy))
prediction=tf.nn.softmax(fc3)
#correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(tfy,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
learning_rate = tf.placeholder(tf.float32, shape=[])
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()


sess.run(init_op)

for i in range(100001):
    indices = np.random.permutation(X_train.shape[0])[:batch_size]
    X_batch = X_train[indices,:,:,:]
    y_batch = y_train[indices,:]
    feed_dict = {tfx : X_batch, tfy : y_batch, learning_rate: 0.04}      
    if i >= 100001: 
        feed_dict = {tfx : X_batch, tfy : y_batch, learning_rate: 0.01}
    
    _, l, predictions = sess.run([train_step, cross_entropy, prediction], feed_dict=feed_dict)

    if (i % 50 == 0):
        print("Iteration: %i. Train loss %.5f, Minibatch accuracy: %.1f%%" % (i,l,accuracy(predictions,y_batch)))
       

for i in range(100001):
    if (i % 50 == 0):
        for j in range(0,X_test.shape[0],batch_size):
            X_batch = X_test[j:j+batch_size,:,:,:]
            y_batch = y_test[j:j+batch_size,:]
            feed_dict = {tfx : X_batch, tfy : y_batch, learning_rate: float(0.04)}      
            if i >= 100001: 
        		feed_dict = {tfx : X_batch, tfy : y_batch, learning_rate: float(0.01)}

            l, predictions = sess.run([cross_entropy, prediction], feed_dict=feed_dict)
            print("Iteration: %i. Test loss %.5f, Minibatch accuracy: %.1f%%" % (i,l,accuracy(predictions,y_batch)))

