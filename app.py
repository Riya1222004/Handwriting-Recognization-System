import streamlit as st
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)#model load
def load_model():
    model=tf.keras.models.load_model('handwritten_character_recog_model.h5')
    return model
model=load_model()

st.write("""
 # HANDWRITING CHARACTER RECAGNITION
""")

file = st.file_uploader("Please upload as image",type=["jpg","png","jpeg"])
import cv2
from PIL import Image,ImageOps
import numpy as np

def predictImage(imageData,model):
    imageData=np.asarray(imageData)
    img = (imageData)[:,:,0]
    img = np.invert(np.array([img]))
    pred= model.predict(img)
    return pred

if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    pred = predictImage(image,model)
    words=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','w','X','Y','Z']
    str= "This character most likely is = "+words[np.argmax(pred)]
    st.success(str)