import pandas as pd
import numpy as np
import PIL
from PIL import Image
import pickle
import random
import matplotlib.pyplot as plt

def save_obj(obj,name):

    image_path = '/Users/admin/Documents/pythonworkspace/data-science-practicum/final-project/gender-age-classification/aligneddicts/'
    with open(image_path+name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

fold_names = ['fold_0_data','fold_1_data','fold_2_data','fold_3_data']
i = 0;
width = 256
height = 256
new_width = 227
new_height = 227

for fold in fold_names:
    df = pd.read_csv('/Volumes/Mac-B/faces-recognition/csvs/'+fold+'.csv')
    inputimages = []
    genders = []
    for index, row in df.iterrows():
        yaw_angle = row['fiducial_yaw_angle']
        gender = row['gender']
        if ((yaw_angle >= -15) and (yaw_angle <= 15) and (gender!='u')):
            if gender:
                folder_name = row['user_id']
                image_name = row['original_image']
                face_id = row['face_id']
            
                image_path = '/Volumes/Mac-B/faces-recognition/aligned/'+folder_name+'/landmark_aligned_face.'+str(face_id)+'.'+image_name
                image = Image.open(image_path)
        
                #Resize image
                image = image.resize((256, 256), PIL.Image.ANTIALIAS)

                image_arr = np.array(image)
                inputimages.append(image_arr)
                genders.append(gender)
                i=i+1

    print (fold)            
    print len(inputimages)            
    print len(genders)
    print ('')

    dict = {'fold_name': fold, 'images': inputimages, 'labels': genders}
    save_obj(dict,fold)
    


'''
            image_arr1 = np.array(image)
            rowseed = random.randint(0,29)
            colseed = random.randint(0,29)
            
            #image_arr2 = image_arr1[rowseed:rowseed+227,colseed:colseed+227,:]
            image_arr2 = image_arr1

            #random lr flipping
            if random.random() < .5:
                print "in ud"
                image_arr2 = image_arr2[::-1,:,:]
                #np.flipud(image_arr2)
                #image_arr2[:,::-1,:]
                
            #random vertical flipping
            if random.random() < .5:
                print "in vertical"
                image_arr2 = image_arr2[:,::-1,:]
                #np.fliplr(image_arr2)
                #
            
            #Center Crop
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2
            image_arr2 = image_arr2[left:right,top:bottom,:]
                

            print image_arr2.shape
            fig = plt.figure()
            a1=fig.add_subplot(1,2,1)
            a2=fig.add_subplot(1,2,2)
            
            a1.imshow(image_arr1)
            a2.imshow(image_arr2)

            plt.show()
'''
        

