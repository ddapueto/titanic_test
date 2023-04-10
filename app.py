from flask import Flask, request, render_template

import pickle
import pandas as pd
import numpy as np
import os

#app = Flask(__name__)


# Initialise the Flask app
app = Flask(__name__, template_folder='templates')


# Use pickle to load in the pre-trained model
with open(f'titanic.pkl', 'rb') as f:
    model = pickle.load(f)

with open('dtypes.pkl', 'rb') as fh:
    dtypes = pickle.load(fh)

with open('pipe.pkl', 'rb') as fh:
    pipeline = pickle.load(fh)


cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
columns_enc = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q',
       'Embarked_S']

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]

    #borro el valor inicial que es el nombre
    name = int_features.pop(0)

    #array
    final = np.array(int_features)

    #dataframe
    data_unseen = pd.DataFrame([final], columns = cols).astype(dtypes)

    #trasnform pipeline
    x_predict = pipeline.transform(data_unseen)

    #dataframe con el pipeline
    X_test_enc = pd.DataFrame(x_predict, columns=columns_enc)

    #prediccion
    prediction = model.predict(X_test_enc)

    if prediction[0] == 0:
        result = 'Morir'
    else:
        result = 'Sobrevivir'

    return render_template('main.html', pred=f'Muchachos, Me parece que {name} si llega a {result}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port)
