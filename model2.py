import numpy as np
import pandas as pd
import os
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns

all_0 = "./C-NMC_Leukemia/training_data/fold_0/all"
all_1 = "./C-NMC_Leukemia/training_data/fold_1/all"
all_2 = "./C-NMC_Leukemia/training_data/fold_2/all"

hem_0 = "./C-NMC_Leukemia/training_data/fold_0/hem"
hem_1 = "./C-NMC_Leukemia/training_data/fold_1/hem"
hem_2 = "./C-NMC_Leukemia/training_data/fold_2/hem"
def get_path_image(folder):
    image_paths = []
    image_fnames = os.listdir(folder)
    for img_id in range(len(image_fnames)):
        img = os.path.join(folder,image_fnames[img_id])
        image_paths.append(img)

    return image_paths

img_data = []

for i in [all_0,all_1,all_2,hem_0,hem_1,hem_2]:
    paths = get_path_image(i)
    img_data.extend(paths)
print(len(img_data))

data = {"img_data":img_data,
        "labels":[np.nan for x in range(len(img_data))]}

data = pd.DataFrame(data)

data["labels"][0:7272] = 1 # ALL
data["labels"][7272:10661] = 0 # HEM

data["labels"] = data["labels"].astype("int64")

img_list = []
for i in range(len(img_data)):
    image = cv.imread(data["img_data"][i])
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    result = cv.bitwise_and(image, image, mask=thresh)
    result[thresh==0] = [255,255,255]
    (x, y, z_) = np.where(result > 0)
    mnx = (np.min(x))
    mxx = (np.max(x))
    mny = (np.min(y))
    mxy = (np.max(y))
    crop_img = image[mnx:mxx,mny:mxy,:]
    crop_img_r = cv.resize(crop_img, (224,224))
    img_list.append(crop_img_r)


from tensorflow.keras.applications import ResNet50, ResNet101
from keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input

def feature_extract(model):
    if model == "VGG19": model = VGG19(weights='imagenet',include_top=False, pooling="avg")
    elif model == "ResNet50": model = ResNet50(weights='imagenet',include_top=False,pooling="avg")
    elif model == "ResNet101": model = ResNet101(weights='imagenet',include_top=False,pooling="avg")
    return model

model = feature_extract("ResNet50") # or "VGG19", "ResNet101" 

features_list = []
for i in range(len(img_list)):

    image = img_list[i].reshape(-1, 224, 224, 3)
    image = preprocess_input(image)

    """
    # Reshaping when VGG19 model is selected
    features = model.predict(image).reshape(512,)
    """

    #Reshaping  when ResNet50 or ResNet101 model is selected
    features = model.predict(image).reshape(2048,)

    features_list.append(features)

features_df = pd.DataFrame(features_list)

features_df["labels"] = data["labels"]

x = features_df.drop(['labels'], axis = 1)
y = features_df.loc[:,"labels"].values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x_ = scaler.transform(x)

x_ = pd.DataFrame(x_)




# testeo de una imagen individual con svm

import cv2 as cv
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import joblib

# Carga el modelo SVM previamente entrenado
loaded_svm_model = joblib.load('svm_model.pkl')

# Carga el modelo ResNet50 para la extracción de características
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling="avg")

# Define una función para preprocesar la imagen de entrada
def preprocess_image(image_path):
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    result = cv.bitwise_and(image, image, mask=thresh)
    result[thresh==0] = [255,255,255]
    (x, y, z_) = np.where(result > 0)
    mnx = (np.min(x))
    mxx = (np.max(x))
    mny = (np.min(y))
    mxy = (np.max(y))
    crop_img = image[mnx:mxx,mny:mxy,:]
    crop_img_r = cv.resize(crop_img, (224, 224))

    return crop_img_r

# Define una función que extrae características de la imagen
def extract_features(image, model, feature_selector):
    image = preprocess_input(image)
    features = model.predict(image.reshape(1, 224, 224, 3))
    selected_features = feature_selector.transform(features.reshape(1, -1))
    return selected_features

# Define una función que clasifica la imagen
def classify_image(image_features, model):
    prediction = model.predict(image_features)

    if prediction[0] == 1:
        return "Clasificación: ALL (Leucemia)"
    else:
        return "Clasificación: HEM (No Leucemia)"

# Función para la selección de características basada en Random Forest
def rf_fs(x, y):
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=619, random_state=5), threshold='1.25*median')
    embeded_rf_selector.fit(x, y)

    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = x.loc[:,embeded_rf_support].columns.tolist()

    rf_x = x[embeded_rf_feature]
    return rf_x

# Ruta de ejemplo para la imagen
user_image_path = "C-NMC_Leukemia/training_data/fold0/hem/UID_H6_8_1_hem.bmp"
ruta = "./C-NMC_Leukemia/training_data/fold_1/hem/UID_H10_3_3_hem.bmp"
# Preprocesa la imagen
user_image = preprocess_image(user_image_path)

# Extrae características de la imagen
image_features = extract_features(user_image, resnet_model, rf_fs(features_df.drop(['labels'], axis=1), features_df['labels']))

# Clasifica la imagen con el modelo SVM previamente entrenado
result = classify_image(image_features, loaded_svm_model)
print(result)
