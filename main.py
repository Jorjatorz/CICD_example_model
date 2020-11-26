# This module defines the entry points to enable the CI/CD framework train and put into production the model

import model as model_package
import pickle

# Adapter for the train, evaluate and predict methods of a model. This will be used by the CI/CD framework to train and deploy the model
# The different methods must be implemented with the designed model
class ModelAdapter():
    # Train method which has as parameter the training dataset to be used
    # This method must return the trained model as a pickle
    def train(self, train_data) -> bytes:
        print("Training model")
        model = model_package.train(train_data)
        binary = None
        try:
            binary = pickle.dumps(model)
        except:
            print("Error while converting the model into a binary")
            return None

        print("Model trained")        
        return binary

    # Evaluation method that, given a model pickle and a test set, should return a dictionary with the different evaluation metrics
    def evaluate(self, model_pickle, test_data) -> dict:
        print("Evaluating model")
        model = pickle.loads(model_pickle)

        score = model_package.evaluate(model, test_data)

        return {"SMAPE": score}

    # Prediction method. Given a set of features, return the moedl prediction.
    # The set of features provided will depende on the type of model and its configuration
    def predict(self, model_pickle, features) -> list:
        print("Predicting")
        model = pickle.loads(model_pickle)
        
        return model_package.predict(model, features)

    # Returns a dict with descriptive information of the model
    def info(self) -> dict:
        model_info = {
            "Name": "Wikipedia webpage visits predictor",
            "Type": "Time series regressor",
            "Evaluation metric": "Symmetric mean absolute percentage error (SMAPE) - The closer to 0 the better.",
            "Description": "This model tries to predict the future visits of Wikipedia's webpage for a given number of days based on past historical data."
        }

        return model_info


# Do not modify this line
entry_point = ModelAdapter()
    