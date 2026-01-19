import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import joblib
import json

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

def train_model():
    file_path = os.path.join("data", "Processed_House_Data.csv")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    # One-hot encoding for location
    dummies = pd.get_dummies(df.location)
    df10 = pd.concat([df, dummies.drop('other', axis='columns')], axis='columns')
    df11 = df10.drop('location', axis='columns')

    X = df11.drop('price', axis='columns')
    y = df11.price

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    # Train a simple linear regression as the primary model
    lr_clf = LinearRegression()
    lr_clf.fit(X_train, y_train)
    print(f"Model Score: {lr_clf.score(X_test, y_test)}")

    # Grid Search for best model
    best_model_results = find_best_model_using_gridsearchcv(X, y)
    print("\nGrid Search Results:")
    print(best_model_results)

    # Save the model and column info
    model_path = os.path.join("models", "home_prices_model.pickle")
    with open(model_path, 'wb') as f:
        joblib.dump(lr_clf, f)
    
    columns = {
        'data_columns': [col.lower() for col in X.columns]
    }
    columns_path = os.path.join("models", "columns.json")
    with open(columns_path, "w") as f:
        f.write(json.dumps(columns))

    print(f"\nModel saved to {model_path}")
    print(f"Columns info saved to {columns_path}")

if __name__ == "__main__":
    train_model()
