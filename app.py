from copyreg import pickle
from crypt import methods
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle

# create flask app
app = Flask(__name__)


# load the pickle model

model = pickle.load(open("model.pkl", "rb"))


# create flask endpoints

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    float_vals = [float(x) for x in request.form.values()]
    vals = [np.array(float_vals)]

    prediction = model.predict(vals)

    return render_template('index.html', prediction=prediction)






if __name__ == '__main__':
    app.run(debug=True)
