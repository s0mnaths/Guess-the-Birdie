
# Guess the Birdie
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/s0mnaths/Guess-the-Birdie/blob/master/notebooks/bird_species_classification.ipynb)




### Classifying Bird Species from bird images using PyTorch

The model architecture used is similar to Resnet9 but smaller and faster. Few convolutional layers with residual layers in between them, along with batch normalization is used. It is trained using PyTorch and then converted to **ONNX** format for easy deplyment using Heroku. For the UI, **Streamlit** has been used.

## Dataset 📂

Dataset used for training is from Kaggle [BIRDS SPECIES IMAGE CLASSIFICATION](https://www.kaggle.com/gpiosenka/100-bird-species) which contains over 58000 training images of more than 300+ species.


## Notebook 📒
View the notebook here: [bird_species_classification.ipynb](https://nbviewer.org/github/s0mnaths/Guess-the-Birdie/blob/master/notebooks/bird_species_classification.ipynb)






## Predictions 🔍
Predictions on unseen test data:


![demo1](https://github.com/s0mnaths/Guess-the-Birdie/blob/master/demo/demo1.png)
![demo2](https://github.com/s0mnaths/Guess-the-Birdie/blob/master/demo/demo2.png)
