import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

Load and Modify the Pre-trained InceptionV3 Model
We use InceptionV3 as a feature extractor by removing the final classification layer.

model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)

Image Preprocessing
Images are resized and normalized to match the input format required by InceptionV3.
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
return x


#Extracting Features

The processed image is passed through the model to obtain a 2048-dimensional feature vector.
def encode_image(img_path):
    img = preprocess_img(img_path)
    feature_vector = model_new.predict(img)
    feature_vector = np.reshape(feature_vector, feature_vector.shape[1])
    return feature_vector

#Uploading Image 
from google.colab import files
uploaded = files.upload()

Dummy Caption Generator (for Demonstration)
def generate_caption(photo_features):
    return "Tiny red roses blooming with quiet elegance amidst a garden of resilience"

#Generate Caption and Display Image
img_path = "plant.jpg"
features = encode_image(img_path)
caption = generate_caption(features)
img = Image.open(img_path)
plt.imshow(img)
plt.title(caption)
plt.axis('off')
plt.show()
