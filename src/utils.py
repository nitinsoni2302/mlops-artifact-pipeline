# src/utils.py
import json #for reading JSON files like config.json
import joblib # for saving/loading machine learning models 
from sklearn.linear_model import LogisticRegression # model we will use
from sklearn.datasets import load_digits # datasetuse
# from sklearn.model_selection import train_test_split # Not directly used here, but common

import os # Used for path manipulations if needed, not strictly for core logic

def load_config(config_path='config/config.json'):
    """
    Loads hyperparameters from a JSON configuration file.
    This function reads the specified JSON file and returns its content as a Python dictionary.
    It also includes basic error handling.
    """
    try:
        with open(config_path, 'r') as f: # It Open the file in read mode ('r')
            config = json.load(f) # it Load JSON content into a Python dictionary
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1) # Exit the script if the config file isn't found
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {config_path}")
        exit(1) # Exit the script if the JSON is invalid

def load_data():
    """
    Loads the digits dataset from scikit-learn.
    The digits dataset is a collection of 8x8 pixel images of handwritten digits (0-9).
    It's a standard dataset for classification tasks.
    """
    digits = load_digits() # it get the dataset
    X = digits.data # X  represents the features (the image data)
    y = digits.target # y  represents the labels (the actual digit value)
    return X, y # Return both features and labels

def create_and_train_model(X, y, config):
    """
    Creates and trains a Logistic Regression model using the provided data and configurations.
    This function takes the features (X), labels (y), and the hyperparameters
    from the config dictionary to initialize and train the model.
    """
    model = LogisticRegression(
        C=config['C'], # Get C from the config
        solver=config['solver'], # Get solver from the config
        max_iter=config['max_iter'], # Get max_iter from the config
        random_state=config.get('random_state', None) # Get random_state (if it exists, otherwise None)
    )
    model.fit(X, y) # This is the training step: the model learns from X and y
    return model # Return the trained model object

def save_model(model, path='model_train.pkl'):
    """
    Saves the trained model to a specified path using joblib.
    Saving the model allows you to reuse it later for predictions without retraining.
    '.pkl' is a common extension for Python pickled (serialized) objects.
    """
    joblib.dump(model, path) # joblib is efficient for saving large NumPy arrays in models
    print(f"Model saved successfully to {path}") # Confirm the save location