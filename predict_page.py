import streamlit as st
from keras.models import load_model
import pandas as pd
import cv2 
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
    col1, col2, col3 = st.columns([1,3,1])

    with col1:
        pass
    with col2:
        st.markdown("<h2 style='text-align: center; color: #83c9ff;'>Number Recognition Model</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: #83c9ff;'>Draw Any Number Between 0 and 9</h4>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: #83c9ff;'>I Bet You I'll Predict it</h4>", unsafe_allow_html=True)
        st.write("")
        canvas_result = st_canvas(
        height=400,
        width=385,
        stroke_width=22,
        stroke_color='#83c9ff',
        background_color='black',
        update_streamlit=True,
        drawing_mode="freedraw",
        key="canvas",
        )
        pred = st.button("Predicit")
        if pred:
            predicit(canvas_result.image_data,col2,col3)
    with col3 :
        pass
         

def predicit(canvas,col2,col3):
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
    
    with col2:
        st.write("")
        st.subheader("The Predicitons is : " + str(np.argmax(pred)))
        st.markdown("<h4 style='text-align: center; color: #83c9ff;'>The Predicitons Probilites in %</h4>", unsafe_allow_html=True)
        st.bar_chart(data=preds, use_container_width=True)
    
   
    