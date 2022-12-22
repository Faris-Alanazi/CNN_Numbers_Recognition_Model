import streamlit as st
from keras.models import load_model
import pandas as pd
import cv2 
import tensorflow 
import pickle
from streamlit_drawable_canvas import st_canvas
import numpy as np


predictions = []
classes = [
"Zero",
"One",
"Two",
"Three",
"Four",
"Five",
"Six",
"Seven",
"Eight",
"Nine",
]

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def load():
    return load_model('model.h5')
cnn_model = load()

def show_page():
    st.title('Number Recognition Model')
    st.subheader("Draw Any Number Between 0 and 9 , I Bet You I'll Predict it") 

    canvas_result = st_canvas(
        fill_color='pink',
        height=500,
        width=700,
        stroke_width=25,
        stroke_color='white',
        background_color='black',
        update_streamlit=True,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    pred = st.button("Predicit",key='btn')

    if pred:
        predicit(canvas_result.image_data)


def predicit(canvas):
    img  = np.array(canvas) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28,28))
    img = img / 255.0
    img = img.reshape(1,28,28,1)
    pred = cnn_model.predict(img)
    predictions , preds = [] , []
    results = [[i,r] for i,r in enumerate(pred[0])]
    results.sort(key=lambda x: x[1], reverse=True)
    for r in results:
        predictions.append([classes[r[0]],float("{:.2f}".format(r[1]*100))])
    preds = pd.DataFrame([x[1] for x in predictions],[x[0] for x in predictions])
    col1, col2 = st.columns([3, 1])
    col1.subheader("The Predicitons Probilites in %")
    col1.bar_chart(data=preds, use_container_width=True)
    col2.subheader("Model Predicitons is")
    col2.header(np.argmax(pred))
   
    
   
    