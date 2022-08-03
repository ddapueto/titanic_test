import pandas as pd
from sklearn.preprocessing import MinMaxScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print('start')

#train
df_train = pd.read_csv('https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/train.csv')

#test
df_test = pd.read_csv('https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/test.csv')

#split target and data
X_train = df_train.drop('Survived', axis=1)
y_train = df_train['Survived']

#delete all columns not necessary
X_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#generar a pipeline para numericas
numeric_transformer = Pipeline(
    steps=[('fill', SimpleImputer(strategy='median')),
           ("scaler", MinMaxScaler())])

#generar un pipeline para categoricas
categorical_transformer = Pipeline(steps=[('fill_cat', SimpleImputer(strategy='most_frequent')),
                                          ("ohe",OneHotEncoder(drop='first', handle_unknown="ignore"))])


#funcion que realiza el trabajo para cada pipeline y luego unirlo
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, make_column_selector(dtype_exclude=object)),
        ("cat", categorical_transformer, make_column_selector(dtype_include = object)),#['Sex', 'Embarked']
    ]
)

preprocessor.fit(X_train)

X_array_enc = preprocessor.transform(X_train)
#genero nombres de columnas

columns_enc = np.append(X_train.select_dtypes(exclude='object').columns, preprocessor.named_transformers_['cat']['ohe'].get_feature_names_out(X_train.select_dtypes(include='object').columns))
X_train_enc = pd.DataFrame(X_array_enc, columns = columns_enc)



model = RandomForestClassifier(n_estimators = 500,verbose=1)
model.fit(X_train_enc, y_train)

import joblib
joblib.dump(preprocessor, 'pipe.joblib')


import pickle

with open('titanic.pkl', 'wb') as file:
  pickle.dump(model, file)

with open('dtypes.pkl', 'wb') as file:
  pickle.dump(X_train.dtypes, file)
