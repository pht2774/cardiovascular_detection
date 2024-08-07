import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

#load the ml model exported from google colab
with open ('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    features = [
        float(request.form['age']),
        float(request.form['sex']),
        float(request.form['cp']),
        float(request.form['trestbps']),
        float(request.form['chol']),
        float(request.form['fbs']),
        float(request.form['restecg']),
        float(request.form['thalach']),
        float(request.form['exang']),
        float(request.form['oldpeak']),
        float(request.form['slope']),
        float(request.form['ca']),
        float(request.form['thal'])
    ]
    
    final_features = [np.array(features)]
    
    # Make the prediction
    prediction = model.predict(final_features)
    
    # Get the result
    output = prediction[0]
    
    return render_template('index.html', prediction_text='Heart Disease Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)