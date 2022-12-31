import streamlit as st
from PIL import Image

#title
st.title("Eye Disease Classification")

#subtitle
st.markdown("## Machine Learning Project")

st.markdown("Link to the app - [eye-disease-classification on 🤗 Spaces](https://huggingface.co/spaces/ameerhamza/eye-disease-classification)")

#image uploader
image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])


# @st.cache
# def load_model(): 
#     reader = ocr.Reader(['en'],model_storage_directory='.')
#     return reader 

# reader = load_model()


if image is not None:

    input_image = Image.open(image) #read image
    st.image(input_image) #display image

    with st.spinner("🤖 AI is at Work! "):        
        label = 'Ameer Hamza'
        st.write(label)
    #st.success("Here you go!")
    st.balloons()
else:
    st.write("Upload an Image")

st.caption("Made by Ameer Hamza, Thanks to Hugging faces")
