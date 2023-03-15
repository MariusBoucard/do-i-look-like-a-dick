from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import joblib
from skimage.io import imread
from skimage.transform import resize
 
	



# modify to fit your system
data_path = fr'{os.getenv("HOME")}/Documents/openclassroom/dataset'
os.listdir(data_path)

base_name = 'data'
width = 224

from PIL import Image
from numpy import asarray
from tensorflow import keras 
  
from keras.utils import load_img,img_to_array



x =[]
y = []
print(data_path)
for dirs in os.walk(data_path):
    for file in dirs[2]:
        # print(dirs[0].split("/")[-1])
        if dirs[0].split("/")[-1] == "dicks" :
         
         
            img = load_img(data_path+"/dicks/"+file, target_size=(224, 224))
            img_array = img_to_array(img)
            # img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            y.append(0)
            x.append(img_array)
         
        if dirs[0].split("/")[-1] == "cat" :
         
            image = Image.open(data_path+"/cat/"+file)
# convert image to numpy array
            
            img = load_img(data_path+"/cat/"+file, target_size=(224, 224))
            img_array = img_to_array(img)
            # img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            y.append(1)
            x.append(img_array)
        if dirs[0].split("/")[-1] == "bald" :
            
            
                img = load_img(data_path+"/bald/"+file, target_size=(224, 224))
                img_array = img_to_array(img)
                # img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0
                
                y.append(2)
                x.append(img_array)

        if dirs[0].split("/")[-1] == "politics" :
            
                img = load_img(data_path+"/politics/"+file, target_size=(224, 224))
                img_array = img_to_array(img)
                # img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0
                
                y.append(3)
                x.append(img_array)

x = np.asarray(x)
y = np.array(y)
print(y)
print(len(y))
print(len(x))

unique, counts = np.unique(y, return_counts=True)
diclass = dict(zip(unique, counts))

print("Voila le nombre d'instance de classes "+str(diclass))
    # image = Image.open(files)
    # # convert image to numpy array
    # data = asarray(image)
    # print(data)
    



# resize_all(src=data_path, pklname=base_name, width=width, include=include)


from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.2, 
    shuffle=True,
    random_state=42,
)
print(X_train[0].shape)
print("set train")



import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


def nn_model():
    model=Sequential()
    model.add(tf.keras.Input(shape=(1,224,224,3)))
    model.add(Dense(224, input_dim=224, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    return model


print(y_train)



seed=10
np.random.seed(seed)

# X_train=X_train.reshape(X_train.shape[0], 1,28,28).astype('float32')
# X_test=X_test.reshape(X_test.shape[0], 1,28,28).astype('float32')

def cnn_model():
    model = Sequential()
    # convolutional layer
    model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(224, 224, 3)))

    # convolutional layer
    model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # flatten output of conv
    model.add(Flatten())

    # hidden layer
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.3))
    # output layer
    model.add(Dense(4, activation='softmax'))
    # model=Sequential()
    # model.add(Conv2D(32,3,3, padding='same',input_shape=(224,224,3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(5,5), padding='same'))
    # model.add(Dropout(0.2))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




X_train=X_train/255
X_test=X_test/255
y_train= np_utils.to_categorical(y_train, num_classes=4)
y_test= np_utils.to_categorical(y_test, num_classes=4)
num_classes=y_train.shape[1]
print(num_classes)

model=cnn_model()
model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=50, batch_size=32  
          )
score= model.evaluate(X_test, y_test, verbose=0)
print('The error is: %.2f%%'%(100-score[1]*100))


##
#
#   TEST
#
#

img = load_img('/home/marius/Documents/openclassroom/dataset/0021.jpg', target_size=(224, 224))
img_array = img_to_array(img)

# remove the extra dimension from the array
img_array /= 255.0
img_array = np.expand_dims(img_array, axis=0)

# pass the image array to the model to make predictions
predictions = model.predict(img_array)

print("predicted dicks"+str(predictions))


img = load_img('/home/marius/Documents/openclassroom/dataset/cat_0007.jpg', target_size=(224, 224))
img_array = img_to_array(img)

# remove the extra dimension from the array
img_array /= 255.0
img_array = np.expand_dims(img_array, axis=0)

# pass the image array to the model to make predictions
predictions = model.predict(img_array)

print("predicted"+str(predictions))


from keras.models import load_model

# train and compile the model...

# save the model to a file
model.save('./model.h5')

img = load_img('/home/marius/Documents/openclassroom/dataset/airplane_0001.jpg', target_size=(224, 224))
img_array = img_to_array(img)

# remove the extra dimension from the array
img_array /= 255.0
img_array = np.expand_dims(img_array, axis=0)

# pass the image array to the model to make predictions
predictions = model.predict(img_array)

print("predicted"+str(predictions))
