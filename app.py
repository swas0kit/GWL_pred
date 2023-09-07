import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    g=request.form.get("g")
    t=request.form.get("t")
    float_features =pd.DataFrame([float(t)],[float(g)]).reset_index()
    print(float_features)
    float_features.rename(columns={'index': 'g',0: 't'}, inplace=True)
    print(float_features)
    prediction = model.predict(float_features)
    return render_template("index.html", prediction_text = "The GWL is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)