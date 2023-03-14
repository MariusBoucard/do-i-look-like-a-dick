from keras.models import load_model
import numpy as np
from keras.utils import load_img, img_to_array
from keras.models import load_model

model = load_model('./model.h5')


img = load_img('/home/marius/Documents/openclassroom/dataset/0021.jpg', target_size=(224, 224))
img_array = img_to_array(img)

# remove the extra dimension from the array
img_array /= 255.0
img_array = np.expand_dims(img_array, axis=0)

# pass the image array to the model to make predictions
predictions = model.predict(img_array)

print("predicted bite"+str(predictions))


img = load_img('/home/marius/Documents/openclassroom/dataset/cat_0007.jpg', target_size=(224, 224))
img_array = img_to_array(img)

# remove the extra dimension from the array
img_array /= 255.0
img_array = np.expand_dims(img_array, axis=0)

# pass the image array to the model to make predictions
predictions = model.predict(img_array)

print("predicted chat"+str(predictions))


from keras.models import load_model



img = load_img('/home/marius/Documents/openclassroom/dataset/us.jpg', target_size=(224, 224))
img_array = img_to_array(img)

# remove the extra dimension from the array
img_array /= 255.0
img_array = np.expand_dims(img_array, axis=0)

# pass the image array to the model to make predictions
predictions = model.predict(img_array)

print("predicted avion :"+str(predictions))

img = load_img('/home/marius/Documents/openclassroom/simba.png', target_size=(224, 224))
img_array = img_to_array(img)

# remove the extra dimension from the array
img_array /= 255.0
img_array = np.expand_dims(img_array, axis=0)

# pass the image array to the model to make predictions
predictions = model.predict(img_array)

print("predicted simba :"+str(predictions))
