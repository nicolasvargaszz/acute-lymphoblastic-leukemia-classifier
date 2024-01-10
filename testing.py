import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize

# Load the model
model = pickle.load(open('img_model.pkl', 'rb'))

Categories = ['hem', 'all']  # Assuming this list is used for classification

def classify_individual_image(image_path):
    # Load and preprocess the image
    img = imread(image_path)
    target_image_size = (100, 100, 3)
    img_resize = resize(img, target_image_size)
    img_array = img_resize.flatten().reshape(1, -1)

    # Predict the category
    probability = model.predict_proba(img_array)
    predicted_category = Categories[model.predict(img_array)[0]]
    return predicted_category

# Replace 'path_to_your_image.jpg' with the actual path to your image
result = classify_individual_image('path_to_your_image.jpg')
print(f"The image is classified as: {result}")
