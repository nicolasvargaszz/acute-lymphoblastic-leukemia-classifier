import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Cargar el modelo entrenado desde el archivo .h5
modelo_cargado = tf.keras.models.load_model('modelokaggle1.h5')  # Reemplaza con la ruta correcta

# Tamaño de entrada esperado por el modelo
input_size = (224, 224)

def cargar_y_preprocesar_imagen(ruta_imagen):
    # Cargar la imagen
    img = image.load_img(ruta_imagen, target_size=input_size)
    img_array = image.img_to_array(img)

    # Agregar una dimensión para el lote (batch)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocesar la imagen
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)

    return img_array

def clasificar_imagen(ruta_imagen):
    # Cargar y preprocesar la imagen
    img_array = cargar_y_preprocesar_imagen(ruta_imagen)

    # Realizar la predicción
    prediction = modelo_cargado.predict(img_array)

    # Obtener la clase predicha (0 si es Benign, 1 si es Early, 2 si es Pre, 3 si es Pro)
    predicted_class = np.argmax(prediction)

    # Mapear la clase predicha a una etiqueta de texto utilizando el diccionario code
    code = {0: "Benign", 1: "Early", 2: "Pre", 3: "Pro"}
    predicted_class_name = code[predicted_class]

    return predicted_class_name

# Ejemplo de clasificación de una imagen individual
imagen_prueba = './Original/Early/WBC-Malignant-Early-084.jpg'  # Reemplaza con la ruta real de tu imagen
resultado = clasificar_imagen(imagen_prueba)
print(f'La imagen se clasifica como: {resultado}')
