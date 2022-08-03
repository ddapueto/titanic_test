#library
import pandas as pd
from sklearn.preprocessing import MinMaxScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import pickle
import joblib


def split_df(data):
    print(f'Start to drop features')
    return data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1) , data['Survived']

def trasnformacion(x_data):

    print(f'Start to transformation')
    #trasnformaciones
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

    x_enc = preprocessor.fit_transform(x_data)

    return  x_enc , preprocessor

def save_pkl(info , name):
    with open(name, 'wb') as file:
        pickle.dump(info, file)

if __name__ == '__main__':
    # train
    df_train = pd.read_csv('https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/train.csv')

    # test
    df_test = pd.read_csv('https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/test.csv')

    #split data to train
    X_train , y_train = split_df(df_train)

    X_train_array_enc , pipe = trasnformacion(X_train)
    columns_enc = np.append(X_train.select_dtypes(exclude='object').columns,
                            pipe.named_transformers_['cat']['ohe'].get_feature_names_out(
                                X_train.select_dtypes(include='object').columns))
    X_train_enc = pd.DataFrame(X_train_array_enc, columns=columns_enc)

    print('start training')
    model = RandomForestClassifier(n_estimators=500)
    model.fit(X_train_enc, y_train)

    for data in [(model , 'titanic.pkl'), (pipe, 'pipe.pkl'), (X_train.dtypes, 'dtypes.pkl')]:
        save_pkl(*data)
