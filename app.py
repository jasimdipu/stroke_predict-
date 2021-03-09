from flask import Flask, render_template, request
import pickle
import sklearn
import jsonify
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('../stroke.pkl', 'rb'))


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/result', methods=['GET'])
def result():
    return render_template('prediction.html')


@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        gender = int(request.form["gender"])
        age_p = float(request.form["age"])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        Residence_type = int(request.form['Residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])

        predict = model.predict(
            [[gender, age_p, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level,
              bmi, smoking_status]])
        return render_template('prediction.html', predict_res=predict)


if __name__ == '__main__':
    app.run()
