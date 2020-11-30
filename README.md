# Example model for the CI/CD Framework
This repository contains an example of a possible model developed and its required interface ```main.py``` required by the [CI/CD Framework](https://github.com/Jorjatorz/ML_CICD_framework).

For the framework to be able to use a model, the model must contain a ```main.py``` file following the same format as ```model_interface_example.py``` (instead of main.py, you can change the name of the used file in ```ML/src/config.py {MODEL_MODULE}```. This class exposes four methods:
1. **Train**: This method gets the dataset as a parameter and should return a pickle of the trained model using this dataset.
2. **Evaluate**: This method accepts the pickle of a trained model and a test dataset (with the same format as the training dataset) and should return a dictionary with the different evaluation metrics of the model's performance.
3. **Predict**: Given the model's pickle and a list of features, this method must return the prediction of the model made with these features.
4. **Info**: This method must return an arbitrary dictionary containing information about the model (name, type, evaluation metrics, description...).

You should import your model's module and implement the four entry points for your model. See this repository main.py for a working example.

The dataset used for developing this example has been downloaded from Kaggle's [Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) by Google. In this competition, the objective is to create a time series predictor for forecasting the number of visits of different Wikipedia.com pages. The evaluation metric used is [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error).

I have chosen this competition as an example because it could be extrapolated to any company seeking to predict its website visits. For the training set, I group all the dataset's pages by date, having the total visits of Wikipedia's webpage per day, from 7/1/2015 to 1/1/2017 (*month/day/year* format). The test set contains the same information but from 1/1/2017 to 2/1/2017.

Hence, the model that is trained, test and deployed in this example (and that can be accessed through the API) tries to predict the future number of daily views that Wikipedia's webpage will have during the following X days (the feature for the predict method) after the 2/1/2017.

A simple Random Forest Regressor is used that, trained using the full training set, gives fairly good results for the website's next month visits.
