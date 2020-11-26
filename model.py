import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# This model is based on the Kaggle competition Web Traffic Time Series Forecasting
# https://www.kaggle.com/c/web-traffic-time-series-forecasting/overview/description

# Given a data frame with the features and target variable, train the model
# See Kaggle competition description for details. Each row is a web page and the different columns are the daily visits to each web page (550 days)
def train(data):
    # Prepare the data
    train_data = _prepare_data(data)
    X = train_data[["weekday", "year", "month", "day", "weekend"]]
    y = train_data["Visits"]

    # Train the model
    model = RandomForestRegressor(n_estimators=10)
    model.fit(X, y)

    return model

# Helper function to convert the original dataset into a row-based dataframe, where each row is the number of visits of a web page in a given day
# These page visits are then aggregated into one row per day, representing the total number of visits of the web page (i.e. Wikipedia) at that day
def _prepare_data(data):
    # Remove missing pages and fill nans with 0 visits
    data = data[data["Page"].notnull()]
    data = data.fillna(0)
    # Transform the temporal date into rows
    train_flattened = pd.melt(data, id_vars='Page', var_name='date', value_name='Visits')
    train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')
    # Aggregate all the views of the different subpages into total daily visits
    train = train_flattened[['Page','date','Visits']].groupby(['date'])['Visits'].sum()
    train = train.reset_index()
    # Transform the date into different features
    _enrich_date(train)

    return train

# Given a data frame with a date column, extract different date-related features from it
def _enrich_date(df):
    df['weekday'] = df['date'].apply(lambda x: x.weekday())
    df['year']=df.date.dt.year 
    df['month']=df.date.dt.month 
    df['day']=df.date.dt.day
    df['weekend'] = ((df.date.dt.dayofweek) // 5 == 1).astype(float)

# Evaluates a model given a test dataset (with the same structure as the original train dataset)
def evaluate(model, test_data):
    test = _prepare_data(test_data)

    X = test[["weekday", "year", "month", "day", "weekend"]]
    y = test["Visits"]

    return smape(model, X, y)

# Predicts the number of visits for the following number of days
def predict(model, num_days):
        # We expect the number of days to predict as feature
        today = "1/2/2017"
        dates = pd.date_range(start=today, periods=num_days)
        X = pd.DataFrame(dates, columns=["date"])
        _enrich_date(X)
        X.drop("date", axis=1, inplace=True)

        pred = None
        try:
            pred = model.predict(X).astype(int)
        except:
            print("Error while performing predictions")

        return list(zip(dates, pred))

# Evaluation function https://www.kaggle.com/c/web-traffic-time-series-forecasting/overview/evaluation
# It is a cost function, the closer to 0 the better
def smape(model, X, y):
    predicted = pd.Series(model.predict(X))

    data = pd.concat([predicted, y], axis=1, keys=['predicted', 'actual'])
    
    evals = abs(data.predicted - data.actual) * 1.0 / (abs(data.predicted) + abs(data.actual)) * 2
    
    return np.sum(evals) / len(data)