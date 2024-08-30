# acute-lymphoblastic-leukemia-classifier
Welcome to Leukemia classifier, this is a proyect where will be using Machine learning, computer vision, and some others tools for the clasification of ALL (acute lymphoblastic leukemia)




Leukemia Classifier with Flask

A Flask web application that leverages machine learning models to classify leukemia images into specific categories.

Overview
This project uses Flask, TensorFlow, and scikit-image to provide a user-friendly interface for classifying leukemia images. It integrates two models: one for distinguishing between 'hem' and 'all' leukemia types and another for classifying 'Benign', 'Early', 'Pre', and 'Pro' leukemia stages.

Features
Image Classification: Upload leukemia images to classify them into specific types and stages.

Model Integration: Leverages machine learning models to deliver accurate classification results.

User Authentication: Secure login functionality for authorized access to diagnostic features.

Setup Instructions
To set up and run this project locally, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/nicolasvargaszz/acute-lymphoblastic-leukemia-classifier.git
cd leukemia-classifier
Install dependencies:


Copy code
pip install -r requirements.txt
Run the Flask app:


Copy code
python app.py
Access the app via http://localhost:5000 in your web browser.

# Usage
### Home/Landing Page: Access the landing page displaying project information and functionality details.

### Login: Authenticate as a doctor by entering valid credentials.

### Upload & Classify: Upload leukemia images to classify them into specific types or stages.

### Results & Database: View classification results and stored data in the app's database.

# Technologies Used



Flask
TensorFlow
scikit-image
SQLite (for the database)

NOTE: 
ALL = acute lymphoblastic leukemia classifier SPA: Leucemia linfoblastica aguda
HEM -> cell without leukemia




![Ejemplo de los resultados:](https://github.com/nicolasvargaszz/acute-lymphoblastic-leukemia-classifier/blob/main/WhatsApp%20Image%202024-07-08%20at%2023.38.07.jpeg)

![Ejemplo de los resultados:](https://github.com/nicolasvargaszz/acute-lymphoblastic-leukemia-classifier/blob/main/WhatsApp%20Image%202024-07-08%20at%2023.27.51.jpeg)

![Ejemplo de los resultados:](https://github.com/nicolasvargaszz/acute-lymphoblastic-leukemia-classifier/blob/main/WhatsApp%20Image%202024-07-08%20at%2023.27.51%20(1).jpeg)
