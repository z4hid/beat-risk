import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from mlflow.models import infer_signature
import os
from urllib.parse import urlparse
import mlflow

def hyperparameter_tuning(X_train, y_train, params):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, params, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

# Load the parameters from params.yaml
params = yaml.safe_load(open('params.yaml'))['train'] 


def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    X=data.drop('HeartDisease',axis=1)
    y=data['HeartDisease']
    
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.set_experiment("heart-disease-prediction")
    # mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"Model saved at {model_path}")


if __name__ == "__main__":
    train(params['input'], params['output'], params['random_state'], params['n_estimators'], params['max_depth'])