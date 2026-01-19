import joblib
import json
import numpy as np
import os

def load_saved_artifacts():
    print("Loading saved artifacts...")
    model_path = os.path.join("models", "home_prices_model.pickle")
    columns_path = os.path.join("models", "columns.json")
    
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    
    with open(columns_path, 'r') as f:
        data_columns = json.load(f)['data_columns']
        
    return model, data_columns

def predict_price(location, sqft, bath, bhk, model, data_columns):
    try:
        loc_index = data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]

if __name__ == "__main__":
    model, data_columns = load_saved_artifacts()
    
    # Sample predictions
    samples = [
        ('1st Phase JP Nagar', 1000, 2, 2),
        ('1st Phase JP Nagar', 1000, 3, 3),
        ('Indira Nagar', 1000, 2, 2),
        ('Indira Nagar', 1000, 3, 3)
    ]
    
    print("\nSample Predictions:")
    for location, sqft, bath, bhk in samples:
        price = predict_price(location, sqft, bath, bhk, model, data_columns)
        print(f"Location: {location}, Sqft: {sqft}, Bath: {bath}, BHK: {bhk} => Estimated Price: {price:.2f} Lakhs")
