import streamlit as st
from PIL import Image
import numpy as np
import joblib
import cv2
import pandas as pd
from sklearn import preprocessing
from skimage.feature import greycomatrix, greycoprops


#title
st.title("Eye Disease Classification")

#subtitle
st.markdown("## Machine Learning Project")

st.markdown("Link to the app - [eye-disease-classification on 🤗 Hugging Faces Spaces](https://huggingface.co/spaces/ameerhamza/eye-disease-classification)")

#image uploader
image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])


#Image preprocessing
def get_eye_cordinate(geteye):
  x_cord = []
  y_cord = []
  w_cord = []
  h_cord = []

  for (x, y, w, h) in geteye:
    x_cord.append(x)
    y_cord.append(y)
    w_cord.append(w)
    h_cord.append(h)

  y_cord.sort()
  y_smallest = y_cord[0]

  h_cord.sort()
  h_largest = h_cord[-1]

  x_cord.sort()
  x_smallest = x_cord[0]

  x_largest = x_cord[-1]

  w_cord.sort()
  w_largest = w_cord[0]
  return y_smallest, h_largest, x_smallest, x_largest, w_largest


def scaleImage(path):
  eyedetect=cv2.CascadeClassifier('https://drive.google.com/file/d/1sDJ-70HBTt_RrNPk4hR07yUiaJDDL35f/view?usp=sharing')
  try:
    image = path
    geteye=eyedetect.detectMultiScale(image)
    y_smallest, h_largest, x_smallest, x_largest, w_largest = get_eye_cordinate(geteye)
    eye_image = image[y_smallest: y_smallest+h_largest, x_smallest:x_largest+w_largest]
    gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)

    return gray
  except Exception as e:
    # count = count+1
    pass


def getAttributes(path):

  eye_image = scaleImage(path)

  from skimage.feature import greycomatrix, greycoprops

  # ----------------- calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ----------------------------------
  def calculation_glcm_all_agls(img, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
      
      glcm = greycomatrix(img, 
                          distances=dists, 
                          angles=agls, 
                          levels=lvl,
                          symmetric=sym, 
                          normed=norm)
      feature = []
      glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
      for item in glcm_props:
              feature.append(item) 
      
      return feature

    # ----------------- call calc_glcm_all_agls() for all properties ----------------------------------
  properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

  glcm_values = []

  glcm = calculation_glcm_all_agls(eye_image, props=properties)
  glcm_values.append(glcm)

  columns_names = []
  angles = ['0', '45', '90','135']
  for name in properties:
      for ang in angles:
          columns_names.append(name + "_" + ang)
        
  return pd.DataFrame(glcm_values, columns=columns_names)


@st.cache
def load_model(): 
    model = joblib.load('https://drive.google.com/file/d/1-2MTwmgF63QI2UC_RkWDU0DYc6GVauyM/view?usp=sharing')
    return model

model = load_model()


if image is not None:

    input_image = Image.open(image) #read image
    st.image(input_image) #display image

    with st.spinner("🤖 AI is at Work! "):        
        label = model.predict(getAttributes(input_image))
        st.write(input_image)
    #st.success("Here you go!")
    st.balloons()
else:
    st.write("Upload an Image")

st.caption("Made by Ameer Hamza, Thanks to Hugging faces")
