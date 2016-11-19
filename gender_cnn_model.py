import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
import sys
import pickle
import random


train_mode = 'age'

def one_hot(y):
    if train_mode == 'gender':
        y_ret = np.zeros((len(y), 2))
    else:
        y_ret = np.zeros((len(y), 8))

    y_ret[np.arange(len(y)), y.astype(int)] = 1
    return y_ret

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


#Read the data from pickles
pickle_file_paths = ['fold_0_data','fold_1_data','fold_2_data','fold_3_data','fold_4_data']
#'fold_1_data','fold_2_data','fold_3_data','fold_4_data']
#pickle_file_path_prefix = '/Volumes/Mac-B/faces-recognition/new_dicts/aligned/'
pickle_file_path_prefix = '/home/ubuntu/gender_age/female/'

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

    if pf == 'fold_4_data':
        images = (pfile['images'])
        #folders_names1 = (pfile['folder_names']) 
        #images_names1 = (pfile['image_names'])
        #faces_ids1 = (pfile['face_ids'])
        
        images = np.array(images)
        X_test.append(images)

        #folders_names1 = np.array(folders_names1)
        #folder_names.append(folders_names1)

        #images_names1 = np.array(images_names1)
        #image_names.append(images_names1)

        #faces_ids1 = np.array(faces_ids1) 
        #face_ids.append(faces_ids1)


    else:

        #dict = {'fold_name': fold, 'images': inputimages, 'labels': genders}
        images = (pfile['images'])

        if train_mode == 'gender':
            gender = (pfile['genders'])

            indices = []
            for i in range(len(gender)):
                if ((gender[i] =='nan') or (gender[i] =='u')):
                    indices.append(i)
            
            images = np.delete(images,indices,axis=0)
            labels = np.delete(gender,indices)

            for i in range(len(labels)):
                if (labels[i] =='None'):
                    indices.append(i)

            images = np.delete(images,indices,axis=0)
            labels = np.delete(labels, indices)

        else:
            labels = (pfile['ages']) 
            indices = []
            
            '''   
            gender = (pfile['genders'])

            for i in range(len(gender)):
                if ((gender[i] =='nan') or (gender[i] =='u')):
                    indices.append(i)

            images = np.delete(images,indices,axis=0)
            labels = np.delete(labels, indices)

            for i in range(len(labels)):
                if (labels[i] =='None'):
                    indices.append(i)

            images = np.delete(images,indices,axis=0)
            labels = np.delete(labels, indices)
            '''
        
        
        images = np.array(images)
        labels = np.array(labels)
        
        labels = one_hot(labels)
        X.append(images)
        y.append(labels)


X = np.array(X)
X = np.vstack(X)

y = np.array(y)
y = np.vstack(y)

X_test = np.array(X_test)
X_test = np.vstack(X_test)

'''
folder_names = np.array(folder_names)
folder_names = np.vstack(folder_names)

image_names = np.array(image_names)
image_names = np.vstack(image_names)

face_ids = np.vstack(face_ids)
face_ids = np.array(face_ids)
'''
print X.shape
print y.shape

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
print "After read all, train shapes: "
print X_train.shape
print y_train.shape

print ("Validation shape: ")
print X_val.shape
print y_val.shape

print "Test shape: "
print X_test.shape


print ('Training, Validation and Test dataset created!!')

train_size = X_train.shape[0]
image_size = 227
num_channels = 3
num_labels=8
batch_size = 50
patch_size = 3
width = 256
height = 256
new_width = 227
new_height = 227


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
w1 = tf.Variable(weight_variable([7,7,3,96]))    
b1 = tf.Variable(bias_variable([96]))
c1 = tf.nn.relu(conv2d(tfx,w1,stride=[1,4,4,1],pad='VALID') + b1)
mxp1 = max_pool(c1,k=[1,3,3,1],stride=[1,2,2,1],pad='VALID')
lrn1 = tf.nn.local_response_normalization(mxp1, alpha=0.0001, beta=0.75)

#Conv Layer 2
w2 = tf.Variable(weight_variable([5,5,96,256]))    
b2 = tf.Variable(bias_variable([256]))
c2 = tf.nn.relu(conv2d(lrn1,w2,stride=[1,1,1,1],pad='SAME') + b2)
mxp2 = max_pool(c2,k=[1,3,3,1],stride=[1,2,2,1],pad='VALID')
lrn2 = tf.nn.local_response_normalization(mxp2, alpha=0.0001, beta=0.75)

#Conv Layer 3
w3 = tf.Variable(weight_variable([3,3,256,384]))    
b3 = tf.Variable(bias_variable([384]))
c3 = tf.nn.relu(conv2d(lrn2,w3,stride=[1,1,1,1],pad='SAME') + b3)
mxp3 = max_pool(c3,k=[1,3,3,1],stride=[1,2,2,1],pad='VALID')

#FC Layer 1
wfc1 = tf.Variable(weight_variable([6 * 6 * 384, 512]))    
bfc1 = tf.Variable(bias_variable([512]))
mxp1_flat = tf.reshape(mxp3, [-1, 6 * 6 * 384])
fc1 = tf.nn.relu(tf.matmul(mxp1_flat, wfc1) + bfc1)
dfc1 = tf.nn.dropout(fc1, 0.5)


#FC Layer 2
wfc2 = tf.Variable(weight_variable([512, 512]))    
bfc2 = tf.Variable(bias_variable([512]))
fc2 = tf.nn.relu(tf.matmul(dfc1, wfc2) + bfc2)
dfc2 = tf.nn.dropout(fc2, 0.7)


#FC Layer 3
wfc3 = tf.Variable(weight_variable([512, num_labels]))  
bfc3 = tf.Variable(bias_variable([num_labels]))
fc3 = (tf.matmul(dfc2, wfc3) + bfc3)
#fc3 = tf.reshape(fc3, [-1, num_labels])
#print "fc3.get_shape"
#print fc3.get_shape


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc3,tfy))

 # L2 regularization for the fully connected parameters.
regularizers = (  tf.nn.l2_loss(wfc3) + tf.nn.l2_loss(bfc3) +
                  tf.nn.l2_loss(wfc2) + tf.nn.l2_loss(bfc2) +
                  tf.nn.l2_loss(wfc1) + tf.nn.l2_loss(bfc1) +
                  tf.nn.l2_loss(w3) + tf.nn.l2_loss(b3) +
                  tf.nn.l2_loss(w2) + tf.nn.l2_loss(b2) +
                  tf.nn.l2_loss(w1) + tf.nn.l2_loss(b1) 
                )

# Add the regularization term to the loss.
cross_entropy += 5e-4 * regularizers

prediction=tf.nn.softmax(fc3)
#correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(tfy,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

learning_rate = tf.placeholder(tf.float32, shape=[])
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
'''
#learning_rate = tf.placeholder(tf.float32, shape=[])
batch = tf.Variable(0, dtype=data_type())
learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * batch_size,  # Current index into the dataset.
      5000,                # Decay step.
      0.0005,              # Decay rate.
      staircase=True)

# Use simple momentum for the optimization.
train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cross_entropy,global_step=batch)
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
'''

# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()


sess.run(init_op)

num_steps = 15000
for i in range(num_steps):
    indices = np.random.permutation(X_train.shape[0])[:batch_size]
    X_batch = X_train[indices,:,:,:]
    y_batch = y_train[indices,:]

    rowseed = random.randint(0,29)
    colseed = random.randint(0,29)
    X_batch = X_batch[:,rowseed:rowseed+227,colseed:colseed+227,:]
    
    #random ud flipping
    if random.random() < .5:
        X_batch = X_batch[:,::-1,:,:]
                
    #random lr flipping
    if random.random() < .5:
        X_batch = X_batch[:,:,::-1,:]
                
    lr = 0.001    
    feed_dict = {tfx : X_batch, tfy : y_batch, learning_rate: lr}      
    if i >= 5001 and i<10001: 
        lr = lr*10
        feed_dict = {tfx : X_batch, tfy : y_batch, learning_rate: lr}
    elif i >= 10001 and i<15001: 
        lr = lr*10
        feed_dict = {tfx : X_batch, tfy : y_batch, learning_rate: lr}   

    _, l, predictions = sess.run([train_step, cross_entropy, prediction], feed_dict=feed_dict)

    if (i % 50 == 0):
        print("Iteration: %i. Train loss %.5f, Minibatch accuracy: %.1f%%" % (i,l,accuracy(predictions,y_batch)))
      
    #validation accuracy
    if (i % 100 == 0):
        val_accuracies = []
        val_losses = []

        for j in range(0,X_val.shape[0],batch_size):
            X_batch = X_val[j:j+batch_size,:,:,:]
            y_batch = y_val[j:j+batch_size,:]

            #Center Crop
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2
            X_batch = X_batch[:,left:right,top:bottom,:]
            
            feed_dict = {tfx : X_batch, tfy : y_batch}   
            l, predictions = sess.run([cross_entropy,prediction], feed_dict=feed_dict)   
            val_accuracies.append(accuracy(predictions,y_batch))
            val_losses.append(l)

        print("Iteration: %i. Val loss %.5f Validation Minibatch accuracy: %.1f%%" % (i, np.mean(val_losses), np.mean(val_accuracies)))
    
    #run model on test
    if (i % 1000 == 0):
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
        np.savetxt('prediction.txt',pred,fmt='%.0f')    
    


