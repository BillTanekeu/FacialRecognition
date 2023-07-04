#import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import pickle 
from functions import histogramme, LBP_function, extract
from PIL import Image
import os
#from tensorflow.image import per_image_standardization

st.set_page_config(page_title = "FaceDetect", page_icon = 'Images/logo.png', layout="wide")
st.image('Images/bant.png')
st.header("Facial recognition using basic LBP and SVM")

model = pickle.load(open('final_model.sav' , 'rb'))





with open("style.css") as f:
    
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html= True)
#st.markdown(" <style> body{background-image: url(image7.jpg);} </style>", unsafe_allow_html= True )
st.sidebar.image("Images/logoFaceDetect.png")
st.sidebar.header("Thank you for using FaceDetect !")
st.sidebar.markdown("<h2> Contact us:</h2>", unsafe_allow_html=True)
c1, c2, c3 = st.sidebar.columns(3)
c1.markdown("<a href =mailto:caterbilljordan.com><img src='https://img.icons8.com/officel/31/000000/gmail-login.png'/> </a> ", unsafe_allow_html=True)
c2.markdown("<a href =https://twitter.com/JTanekeu><img src='https://img.icons8.com/color/31/000000/twitter--v1.png'/></a> ", unsafe_allow_html=True)
c3.markdown("<a href =https://t.me/Tanekeu> <img src='https://img.icons8.com/color/31/000000/telegram-app--v1.png'/> </a> ", unsafe_allow_html=True)



st.snow()
def Main():

    file = st.file_uploader("Choose a picture", type = [ 'PGM','pgm'])
    if file is not None:
        save_uploaded_file(file)
        image = Image.open(file).convert('L')
        figure = plt.figure(figsize = (5,5))
        plt.imshow(image)
        plt.axis("off")
        result = prediction('uploadedfile.pgm', model)
        col1, col2= st.columns(2)
        col1.metric("Sujet:", file.name[:-4] )
        col2.metric("Reconnaissance: ", result)
        st.pyplot(figure)





def save_uploaded_file(uploadedfile):
    with open(os.path.join(os.getcwd(), 'uploadedfile.pgm'), "wb") as f:
        f.write(uploadedfile.getbuffer())



def prediction(image, model, window = 10):
    data = extract(image)
    lbp = LBP_function(data)
    histo = histogramme(lbp , window = 10)
    histo = [histo]
    rst = model.predict(histo)
    return rst


Main()