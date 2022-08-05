import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)

def load_model():
    model=tf.keras.models.load_model('cifar10_cnn (2).h5')

    return model

with st.spinner('Model is being loaded ...'):
    model=load_model()
    
st.write("""
# cifar10 classification app""")
st.write('-----------')
file =st.file_uploader("please upload an image",type=["jpg","png"])
# handling in image
import cv2
from PIL import Image,ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding',False)

def import_and_predict(image_data,model):
    size=(32,32)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    image=np.asarray(image)
    img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction

if file is None:
    st.text('please upload an image file')
    
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    predicitions=import_and_predict(image,model)
    score=tf.nn.softmax(predicitions[0])
    st.write(predicitions)
    st.write(score)
    classes_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    
    st.write('this image belongs to {} with a {:.2f} percent confidence'.format(classes_names[np.argmax(score)],100*np.max(score)))
    
    
   
