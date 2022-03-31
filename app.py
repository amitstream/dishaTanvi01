# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import streamlit as st
import requests
import json

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


def get_prediction_pheno(data={"SITE_ID":"ABIDEII-KKI_1","SUB_ID":29554,"AGE_AT_SCAN":34.564,"SEX":1,"HANDEDNESS_CATEGORY":1,"HANDEDNESS_SCORES":0,"FIQ":100,"VIQ":0,"PIQ":0,"FIQ_TEST_TYPE":"WISC-IV","SRS_EDITION":1,"SRS_VERSION":1,"SRS_INFORMANT":1,"SRS_TOTAL_RAW":81,"SRS_AWARENESS_RAW":10.5,"SRS_COGNITION_RAW":15.5,"SRS_COMMUNICATION_RAW":27.5,"SRS_MOTIVATION_RAW":14.5,"SRS_MANNERISMS_RAW":18,"SRS_TOTAL_T":73.5,"SRS_AWARENESS_T":65,"SRS_COGNITION_T":71,"SRS_COMMUNICATION_T":73,"SRS_MOTIVATION_T":71.5,"SRS_MANNERISMS_T":89,"CURRENT_MED_STATUS":0}):
  url = 'https://askai.aiclub.world/c22776b6-21e7-4a55-bbf4-444e6fa6c7b5'
  print(data)
  r = requests.post(url, data=json.dumps(data))
  response = getattr(r,'_content').decode("utf-8")
  print(response)
  j=json.loads(response)
  b=j["body"]
  print("B=",b)
  j2=json.loads(b)
  p=j2["predicted_label"]
  c=j2["confidence_score"]
  c1=c['1']
  c2=c['2']
  print("P=",p," while C=",c, " with c1=",c1," and c2=",c2)
  return p,c1,c2

def processImageFile(f):
    st.image(f)
    # Please replace the brackets below with the location of your image which need to predict
    img = Image.open(f)
    img = img.resize(IMG_SIZE, Image.ANTIALIAS)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    prediction_sample = new_model.predict(img_preprocessed)
    score_1 = prediction_sample[0][0]
    score_2 = prediction_sample[0][1]
    
    if (score_1 >= score_2):
       predicted_label = 1
    else:
      predicted_label = 2
    print("\nIMAGE processing:",predicted_label,score_1, score_2)
    return predicted_label,score_1,score_2
#
def combineResults(r1,c1,c2,r2,i1,i2):
  s=f'Combined results: {r1},{c1},{c2},{r2},{i1},{i2}'
  print("Combined results: ",r1,c1,c2,r2,i1,i2)
  st.title(s)
#
def uploadFiles():
  st.title("Patient information")
  uploadedCsvFile=st.file_uploader("Choose patient record")
  uploadedImageFile=st.file_uploader("Choose MRI file")
  if st.button("Run AI"):
    if uploadedCsvFile is not None:
      df = pd.read_csv(uploadedCsvFile)
      records= df.to_dict(orient='records')
      resp,c1,c2=get_prediction_pheno(data=records[0])
      print("Pheno resp:",resp," and confidence=",c1," and ",c2)
    if uploadedImageFile is not None:
      (l,s1,s2)= processImageFile(uploadedImageFile)
    combineResults(resp,c1,c2,l,s1,s2)
# Main
uploadFiles()

