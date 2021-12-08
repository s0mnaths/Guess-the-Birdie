import streamlit as st
from PIL import Image
from classify import classifyImages, species_list


st.title("Bird Species Classification")
st.text("Upload an image of a bird to know it's species")
uploaded_file = st.file_uploader("Upload an image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image.', use_column_width=True)
    st.write("")
    st.write("The bird species is:")
    label = classifyImages(image, 'weights91.pth')
    species = species_list.iloc[label, 1]
    st.header(species)