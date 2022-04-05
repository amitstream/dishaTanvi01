import numpy as np
import pandas as pd
import os

from tensorflow.keras.preprocessing import image
from keras.applications.resnet import preprocess_input
from PIL import Image

import tensorflow as tf
import streamlit as st
import requests
import json

IMG_SIZE = (224, 224)

topTitle=st.empty()
imageInputShow=st.empty()
fileInputShow=st.empty()
imageAIShow=st.empty()
fileAIShow=st.empty()

#
def get_prediction_pheno(data):
  topTitle.title("Processing user data")
  url = 'https://askai.aiclub.world/c22776b6-21e7-4a55-bbf4-444e6fa6c7b5'
  r = requests.post(url, data=json.dumps(data))
  response = getattr(r,'_content').decode("utf-8")
  #print(response)
  j=json.loads(response)
  b=j["body"]
  j2=json.loads(b)
  p=j2["predicted_label"]
  c=j2["confidence_score"]
  c1=c['1']
  c2=c['2']
  print("P=",p," while C=",c, " with c1=",c1," and c2=",c2)
  fileAIShow.title(f'Pheno {p}, {c1},{c2}')
  return p,c1,c2
#
def processImageFile(f):
  topTitle.title("Processing MRI image")
  if "model_loaded" not in st.session_state:
    #new_model=load_my_model('best_model_full_vac_0.66')
    new_model=tf.keras.models.load_model('best_model_full_vac_0.66')
    st.session_state.model_loaded=new_model
  else:
    new_model=st.session_state.model_loaded
  # Please replace the brackets below with the location of your image which need to predict
  img = Image.open(f)
  img = img.resize(IMG_SIZE, Image.ANTIALIAS)
  img = img.convert(mode="RGB")
  img_array = image.img_to_array(img)
  img_batch = np.expand_dims(img_array, axis=0)
  img_preprocessed = preprocess_input(img_batch)
  #new_model=load_my_model('best_model_full_vac_0.66')
  prediction_sample = new_model.predict(img_preprocessed)
  score_1 = prediction_sample[0][0]
  score_2 = prediction_sample[0][1]
    
  if (score_1 >= score_2):
    predicted_label = 1
  else:
    predicted_label = 2
  print("\nIMAGE processing:",predicted_label,score_1, score_2)
  imageAIShow.title(f'Image: {predicted_label},{score_1},{score_2}')
  return predicted_label,score_1,score_2
#
def combineResults(r1,c1,c2,r2,i1,i2):
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
#
def uploadFiles():
  global uploadedCsvFile,uploadedImageFile
  topTitle.title("Getting patient information")
  uploadedCsvFile=st.sidebar.file_uploader("Choose patient record")
  uploadedImageFile=st.sidebar.file_uploader("Choose MRI file")
  if uploadedCsvFile is not None:
    df = pd.read_csv(uploadedCsvFile)
    records= df.to_dict(orient='records')
    fileInputShow.write(records)
  if uploadedImageFile is not None:
    imageInputShow.image(uploadedImageFile)
  if st.sidebar.button("Run AI"):
    if uploadedCsvFile is not None:
      resp,c1,c2=get_prediction_pheno(data=records[0])
      print("Pheno resp:",resp," and confidence=",c1," and ",c2)
    if uploadedImageFile is not None:
      (l,s1,s2)= processImageFile(uploadedImageFile)
    combineResults(resp,c1,c2,l,s1,s2)
# Main
uploadFiles()

