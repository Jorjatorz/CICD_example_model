# This module defines the entry points to enable the CI/CD framework train and put into production the model.
# A main.py file must be created and this class must be implemented with the desired model.

import model as model_package
import pickle

# Adapter for the train, evaluate and predict methods of a model. This will be used by the CI/CD framework to train and deploy the model
# The different methods must be implemented with the designed model
class ModelAdapter():
    # Train method which has as parameter the training dataset to be used
    # This method must return the trained model as a pickle
    def train(self, train_data) -> bytes:
        print("Training model")
        # Your training code here

    # Evaluation method that, given a model pickle and a test set, should return a dictionary with the different evaluation metrics
    def evaluate(self, model_pickle, test_data) -> dict:
        print("Evaluating model")
        # Your evaluation code here

    # Prediction method. Given a set of features, return the moedl prediction.
    # The set of features provided will depende on the type of model and its configuration
    def predict(self, model_pickle, features) -> list:
        print("Predicting")
        # Your prediction code here

    # Returns a dict with descriptive information of the model
    def info(self) -> dict:
        # You can populate this dictionary with any entry you may find useful
        model_info = {
            "Name": "XXX",
            "Type": "XXX",
            "Evaluation metric": "XXX",
            "Description": "XXX"
        }

        return model_info


# Do not modify this line
entry_point = ModelAdapter()
    