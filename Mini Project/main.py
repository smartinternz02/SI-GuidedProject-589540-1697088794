import pickle
from flask import Flask, render_template , request
import pandas as pd
import numpy as np

# Create Flask app
model = pickle.load(open("C:/Users/zeesh/OneDrive/Desktop/mini project/templates/bodyfat.pkl", "rb"))
app = Flask(__name__)

app.static_folder = "static"

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict')
def index():
    return render_template('predict.html')

@app.route('/data_predict', methods=['GET', 'POST'])
def predict() :
        density = float(request.form['Density'])
        age = float(request.form['Age'])
        height = float(request.form['Height'])
        neck = float(request.form['Neck'])
        ankle = float(request.form['Ankle'])
        biceps = float(request.form['Biceps'])
        forearm = float(request.form['Forearm'])
        wrist = float(request.form['Wrist'])

        prediction = model.predict(pd.DataFrame([[density, age, height, neck, ankle, biceps, forearm, wrist]], columns=['Density', 'Age', 'Height', 'Neck', 'Ankle', 'Biceps', 'Forearm', 'Wrist']))

        prediction = np.round(prediction, 2)

        return render_template('fat_predict.html', prediction=prediction[0])
        
if __name__ == "__main__":
    app.run(debug=True)