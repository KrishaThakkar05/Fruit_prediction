import streamlit as st
import joblib
import cv2
import os
import numpy as np
st.title("Fruit Prediction(Apple/Mango)")
model=joblib.load("model.pkl")
st.write("Upload Image")
file=st.file_uploader("Here,Upload File")
if file is not None:
    img=np.frombuffer(file.read(),np.uint8)
    img=cv2.imdecode(img,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img)
    img=cv2.resize(img,(32,32))
    img=img.flatten()
    img=img.reshape(1,-1)/255.0
    pred=model.predict(img)
    if pred[0]==0:
        st.write("Fruit Is Apple")
    else:
        st.write("Fruit Is Mango")   