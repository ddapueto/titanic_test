from flask import Flask, request, render_template
import joblib
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


# Initialise the Flask app
app = Flask(__name__, template_folder='templates')


# Use pickle to load in the pre-trained model
with open(f'titanic.pkl', 'rb') as f:
    model = pickle.load(f)

with open('dtypes.pkl', 'rb') as fh:
    dtypes = pickle.load(fh)

with open('pipe.pkl', 'rb') as fh:
    pipeline = pickle.load(fh)

#pipeline = joblib.load('pipe.joblib')
cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
columns_enc = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q',
       'Embarked_S']

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    print('hola')
    name = int_features.pop(0)

    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols).astype(dtypes)

    x_predict = pipeline.transform(data_unseen)
    X_test_enc = pd.DataFrame(x_predict, columns=columns_enc)

    prediction = model.predict(X_test_enc)

    if prediction[0] == 0:
        result = 'Morir'
    else:
        result = 'Sobrevivir'

    return render_template('main.html', pred=f'Muchachos, Me parece que {name} si llega a {result}')

if __name__ == '__main__':
    app.run()
