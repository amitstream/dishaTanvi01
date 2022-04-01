# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import tensorflow as tf
import streamlit as st
import requests
import json
import datetime

from keras.applications.resnet import preprocess_input
#from keras.preprocessing.image import ImageDataGenerator
#from keras.layers import Dense,GlobalAveragePooling2D
#from keras.models import Model
#from tensorflow.keras import regularizers

from PIL import Image

#from tensorflow.keras.preprocessing import image_dataset_from_directory
#from keras.callbacks import EarlyStopping
#from tensorflow import keras

#
# Used to be @ ST  st.cache
# Used to be @ ST  st.cache  (hash_funcs={"MyUnhashableClass": lambda _: None})
# Used to be @ ST st.experimental_memo
def load_my_model(f='best_model_full_vac_0.66'):
  global core_model
  core_model=tf.keras.models.load_model(f)
  return core_model

print("At Step 1 of program",datetime.datetime.now())

# Predicting code for an image
IMG_SIZE = (224, 224)
from tensorflow.keras.preprocessing import image
print("At Step 2 of program",datetime.datetime.now())
topTitle=st.empty()
imageShow=st.empty()
fileShow=st.empty()
#
def get_prediction_pheno(data):
  topTitle.title("Processing user data")
  print("At Step 1 of get_prediction_pheno",datetime.datetime.now())
  url = 'https://askai.aiclub.world/c22776b6-21e7-4a55-bbf4-444e6fa6c7b5'
  print(data)
  r = requests.post(url, data=json.dumps(data))
  print("At Step 2 of get_prediction_pheno",datetime.datetime.now())
  response = getattr(r,'_content').decode("utf-8")
  print("At Step 3 of get_prediction_pheno",datetime.datetime.now())
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
  print("At Step 5 of get_prediction_pheno",datetime.datetime.now())
  return p,c1,c2
#
def noOp():
  print("Called No-op",datetime.datetime.now())
#
def processImageFile(f):
  print("At Step 1 of process_image_file",datetime.datetime.now())
  topTitle.title("Processing MRI image")
  if "model_loaded" not in st.session_state:
    print("Loading session state at first invocation of processImageFile",datetime.datetime.now())
    #new_model=load_my_model('best_model_full_vac_0.66')
    new_model=tf.keras.models.load_model('best_model_full_vac_0.66')
    st.session_state.model_loaded=new_model
    print("Finished loading session state at first invocation of processImageFile",datetime.datetime.now())
  else:
    print("Found cached model in session state",datetime.datetime.now())
    new_model=st.session_state.model_loaded
  print("At Step 2 of process_image_file",datetime.datetime.now())
  # Please replace the brackets below with the location of your image which need to predict
  img = Image.open(f)
  img = img.resize(IMG_SIZE, Image.ANTIALIAS)
  img_array = image.img_to_array(img)
  img_batch = np.expand_dims(img_array, axis=0)
  img_preprocessed = preprocess_input(img_batch)
  print("At Step 3 of process_image_file",datetime.datetime.now())
  #new_model=load_my_model('best_model_full_vac_0.66')
  print("At Step 4 of process_image_file",datetime.datetime.now())
  prediction_sample = new_model.predict(img_preprocessed)
  print("At Step 5 of process_image_file",datetime.datetime.now())
  score_1 = prediction_sample[0][0]
  score_2 = prediction_sample[0][1]
    
  if (score_1 >= score_2):
    predicted_label = 1
  else:
    predicted_label = 2
  print("\nIMAGE processing:",predicted_label,score_1, score_2)
  print("At Step 7 of process_image_file",datetime.datetime.now())
  return predicted_label,score_1,score_2
#
def combineResults(r1,c1,c2,r2,i1,i2):
  print("At Step 1 of combine_results",datetime.datetime.now())
  s=f'Combined results: {r1},{c1},{c2},{r2},{i1},{i2}'
  print("Combined results: ",r1,c1,c2,r2,i1,i2)
  if(r1==1 and r2==1):
    resp="Both AIs agree Conclusion 1"
  elif(r1==2 and r2==2):
    resp="Both AIs agree Conclusion 2"
  else: 
    if(c1+i1>c2+i2):
      resp="Disagreement, but majority favors 1"
    else:
      resp="Disagreement, but majority favors 2"
  topTitle.title(resp)
  print("At Step 7 of combine_results",datetime.datetime.now())
#
def runAI():
  print("At Step 1 of runAI",datetime.datetime.now())
  if uploadedCsvFile is not None:
    df = pd.read_csv(uploadedCsvFile)
    records= df.to_dict(orient='records')
    resp,c1,c2=get_prediction_pheno(data=records[0])
    print("Pheno resp:",resp," and confidence=",c1," and ",c2)
  print("At Step 7 of runAI",datetime.datetime.now())
  if uploadedImageFile is not None:
    (l,s1,s2)= processImageFile(uploadedImageFile)
    print("At Step 8 of runAI",datetime.datetime.now())
    combineResults(resp,c1,c2,l,s1,s2)
  print("At Step 9 of runAI",datetime.datetime.now())
#
def uploadFiles():
  global uploadedCsvFile,uploadedImageFile
  print("At Step 1 of upload_files",datetime.datetime.now())
  topTitle.title("Waiting for patient information")
  uploadedCsvFile=st.sidebar.file_uploader("Choose patient record")
  uploadedImageFile=st.sidebar.file_uploader("Choose MRI file")
  print("At Step 3 of upload_files",datetime.datetime.now())
  if uploadedCsvFile is not None:
    df = pd.read_csv(uploadedCsvFile)
    records= df.to_dict(orient='records')
    fileShow.write(records)
  if uploadedImageFile is not None:
    imageShow.image(uploadedImageFile)
  if st.sidebar.button("Run AI"):
    print("At Step 5 of upload_files",datetime.datetime.now())
    if uploadedCsvFile is not None:
      resp,c1,c2=get_prediction_pheno(data=records[0])
      print("Pheno resp:",resp," and confidence=",c1," and ",c2)
    print("At Step 7 of upload_files",datetime.datetime.now())
    if uploadedImageFile is not None:
      (l,s1,s2)= processImageFile(uploadedImageFile)
    print("At Step 8 of upload_files",datetime.datetime.now())
    combineResults(resp,c1,c2,l,s1,s2)
  print("At Step 9 of upload_files",datetime.datetime.now())
# Main
uploadFiles()

