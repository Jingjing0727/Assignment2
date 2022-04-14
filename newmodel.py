import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm
from sklearn.pipeline import Pipeline

import joblib

import matplotlib.pyplot as plt
#data
df_train = "train1.csv"
df_test = "test1.csv"

#X,y
def load_prepare():
    df = pd.read_csv(df_train)
    X = df.drop(columns=['income_>50K'])
    y = df['income_>50K']
    return X, y

#final model:
def build_pipeline_final(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    pipeline = Pipeline([('sc', StandardScaler()),
    ('RFC', RandomForestClassifier(max_depth = 6,random_state=300))])
    pipeline.fit(X_train, y_train)
    y_predict = pipeline.predict(X_test)
    training_accuracy = accuracy_score(y_test, y_predict)
    confusion_matrix = cm(y_test, y_predict)
    joblib.dump(pipeline, 'pipeline.pkl')
    # return training accuracy, sklearn confusion matrix (from validation step) and sklearn pipeline object
    return training_accuracy, confusion_matrix, pipeline

def apply_pipeline():
    pipeline = joblib.load('pipeline.pkl')
    df = pd.read_csv(df_test)
    predictions = pipeline.predict(df)
    # return predictions or outcomes
    return predictions


