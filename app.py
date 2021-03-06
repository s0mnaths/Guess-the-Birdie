import streamlit as st
from PIL import Image
from classify.classify import classifyImages, species_list
from classify.detect import detectBird


st.title("Bird Species Classification")
st.text("Upload an image of a bird to know it's species")
uploaded_file = st.file_uploader("Upload an image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image.', use_column_width=True)
    
    birdImg = detectBird(image)
    st.write("The bird species is:")
    label = classifyImages(birdImg)
    species = species_list.iloc[label, 1]
    st.header(species)