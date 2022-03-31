# import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import streamlit as st

from keras.applications.resnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras import regularizers

from PIL import Image

from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.callbacks import EarlyStopping
from tensorflow import keras

new_model = tf.keras.models.load_model('best_model_full_vac_0.66')
# Predicting code for an image
IMG_SIZE = (224, 224)
from tensorflow.keras.preprocessing import image
def processFile(uploadedFile):
    st.image(uploadedFile)
    # Please replace the brackets below with the location of your image which need to predict
    img = Image.open(uploadedFile)
    img = img.resize(IMG_SIZE, Image.ANTIALIAS)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    prediction_sample = new_model.predict(img_preprocessed)
    score_1 = prediction_sample[0][0]
    score_2 = prediction_sample[0][1]
    
    print(score_1, score_2)
    if (score_1 >= score_2):
       predicted_label = 1
    else:
      predicted_label = 2
    return predicted_label,score_1,score_2
def uploadAI():
  st.title("File upload")
  uploadedFile=st.file_uploader("Choose file")
  if uploadedFile is not None:
    (l,s1,s2)= processFile(uploadedFile)
    return l
# Main
r=uploadAI()
st.title(r)

