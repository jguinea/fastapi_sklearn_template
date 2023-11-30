import pandas as pd
from decouple import config
import os
from helper.utils import  save_obj, load_obj
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def get_model(X):
    categorical_features = list(X.dtypes[X.dtypes == 'category'].index)
    numerical_features = list(X.dtypes[X.dtypes != 'category'].index)
    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    numerical_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])

    vector_cleaning = ColumnTransformer(
        [
            ("cat", categorical_encoder, categorical_features),
            ("num", numerical_pipe, numerical_features),
        ]
    )

    pca = PCA()
    preprocessing = Pipeline(
        [("vector_cleaning", vector_cleaning),
        ("dim_reduction", pca),]
    )
    model = RandomForestClassifier()
    pipeline = Pipeline([("preprocessing", preprocessing), ("classifier", model)])

class MyModel:
    def __init__(self, trained=False):
        self.trained = trained
        self.X = pd.read_csv(os.path.join(config("DATA_PATH"),"X.csv"), index_col=0)
        self.y = pd.read_csv(os.path.join(config("DATA_PATH"),"y.csv"), index_col=0)
        if self.trained:
            self.pipe = load_obj("pipeline")
        else:
            model=get_model(self.X).set_params(config("pipeline_params"))
        return
    
    def train_model(self):
        if (not self.pretrained):
            self.pipe.fit(self.X,self.y)
            self.trained = True
            return
        else:
            return
    
    def predict_label(self, y_new):
        y_pred = self.pipe.predict(y_new)
        return y_pred
    
    def predict_labels(self, y_new):
        y_pred = self.pipe.predict(y_new)
        return y_pred