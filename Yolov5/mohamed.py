import streamlit as st
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
img = 'data/images/zidane.jpg'
results = model(img)
file=st.file_uploader('please upload image',type=['jpg','png'])
image=Image.open(file)

    

    
