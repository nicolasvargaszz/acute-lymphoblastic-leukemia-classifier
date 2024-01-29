from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from datetime import datetime



app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///results.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db) 

Categories = ['hem', 'all']

# Load the different model for 'upload()' function
model = pickle.load(open('img_model.pkl', 'rb')) # ESTE ES EL MODELO que clasifica HEM O ALL

# Load the original model 'modelo_cargado' for 'upload_diagnostico()' function
modelo_cargado = tf.keras.models.load_model('modelokaggle1.h5')  # Este es el modelo que clasifica pre, pro, early, etc.
input_size = (224, 224)

# Define the model for the classification results
class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_first_name = db.Column(db.String(100))
    patient_last_name = db.Column(db.String(100))
    image_path = db.Column(db.String(200))
    classification_result = db.Column(db.String(50))
    registration_date = db.Column(db.DateTime,default=datetime.utcnow)


def classify_image_with_model(image_path):
    img = imread(image_path)
    target_image_size = (100, 100, 3)
    img_resize = resize(img, target_image_size)
    img_array = img_resize.flatten().reshape(1, -1)

    probability = model.predict_proba(img_array)
    predicted_category = Categories[model.predict(img_array)[0]]
    return predicted_category


def cargar_y_preprocesar_imagen(ruta_imagen):
    img = image.load_img(ruta_imagen, target_size=input_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
    return img_array

def clasificar_imagen(ruta_imagen):
    img_array = cargar_y_preprocesar_imagen(ruta_imagen)
    prediction = modelo_cargado.predict(img_array)
    predicted_class = np.argmax(prediction)
    code = {0: "Benign", 1: "Early", 2: "Pre", 3: "Pro"}
    predicted_class_name = code[predicted_class]
    return predicted_class_name


@app.route('/')
def landing_page():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'Doctor' and password == '2104': #si la contra esta god
            return redirect(url_for('upload_diagnostico'))
        else: # si esta mal
            return "Invalid credentials. Please try again."
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' in request.files:
            uploaded_file = request.files['image']
            if uploaded_file.filename != '':
                image_path = os.path.join('temp', uploaded_file.filename)
                uploaded_file.save(image_path)

                result = classify_image_with_model(image_path)

                current_date = datetime.utcnow()
                os.remove(image_path)
                new_result = Result(
                    patient_first_name=request.form['first_name'],
                    patient_last_name=request.form['last_name'],
                    image_path=image_path,
                    classification_result=result,
                    registration_date=current_date
                )
                db.session.add(new_result)
                db.session.commit()

                if result in ['hem', 'all']:
                    return redirect(url_for('result_hem_all', result=result))
                else:
                    return redirect(url_for('upload_diagnostico', result=result))
    return render_template("upload_data.html")

@app.route('/result_diagnostico', methods=['GET'])
def show_result_diagnostico():
    result = request.args.get('result')
    return render_template('result_diagnostico.html', result=result)

@app.route('/result_hem_all', methods=['GET'])
def result_hem_all():
    result = request.args.get('result')
    return render_template('result_hem_all.html', result=result)

@app.route('/database', methods=['GET'])
def database():    
    results = Result.query.all()
    return render_template('database.html', results=results)

@app.route('/upload_diagnostico', methods=['GET', 'POST'])
def upload_diagnostico():
    if request.method == 'POST':
        if 'image' in request.files:
            uploaded_file = request.files['image']
            if uploaded_file.filename != '':
                image_path = os.path.join('temp', uploaded_file.filename)
                uploaded_file.save(image_path)

                result_diagnostico = clasificar_imagen(image_path)

                os.remove(image_path)
                new_result = Result(
                    patient_first_name=request.form['first_name'],
                    patient_last_name=request.form['last_name'],
                    image_path=image_path,
                    classification_result=result_diagnostico
                )
                db.session.add(new_result)
                db.session.commit()

                if result_diagnostico in ['Benign', 'Early', 'Pre', 'Pro']:
                    return redirect(url_for('show_result_diagnostico', result=result_diagnostico))
    return render_template("upload_data_diagnostico.html")

if __name__ == '__main__':
    app.run(debug=True)