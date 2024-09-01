import numpy as np
from numpy.linalg import norm
import os
import pickle

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3)) # This line is unchanged
model.trainable=False # This line is unchanged

# Create the Sequential model and add an input layer with the desired shape
model=tensorflow.keras.Sequential([
    tensorflow.keras.layers.InputLayer(input_shape=(224,224,3)), # Add this Input layer
    model,
    GlobalMaxPooling2D()
])
model.summary()

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

print(os.listdir('D:/Data Science/Project/ten/newenv/myntradataset/images'))


filenames=[]
for file in os.listdir('D:/Data Science/Project/ten/newenv/myntradataset/images'):
  filenames.append(os.path.join('D:/Data Science/Project/ten/newenv/myntradataset/images',file))

  print(len(filenames))

  from tqdm import tqdm

  feature_list=[]
for file in tqdm(filenames):
  feature_list.append(extract_features(file,model))
print(np.array(feature_list).shape)

import pickle
pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))
