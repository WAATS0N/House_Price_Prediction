# House Price Prediction

A machine learning project designed to estimate residential property prices in Bangalore, India. This tool uses historical house data, performs data cleaning, explores features, and trains predictive models (Linear Regression, Lasso, Decision Trees) to provide accurate price estimates based on location, square footage, and property size.

## Features

- **Data Preprocessing**: Handles missing values, performs feature engineering (e.g., total_sqft conversion), and outlier detection.
- **Model Training**: Implements multiple regression algorithms with automated hyperparameter tuning using `GridSearchCV`.
- **Price Prediction**: Provides a simple interface to estimate house prices for specific locations and configurations.
- **Extensible**: Structured setup for adding new models or more complex preprocessing steps.

## Project Structure

```text
House_Price_Prediction/
├── data/               # Raw and processed datasets (CSV)
├── models/             # Saved model artifacts (.pickle, .json)
├── notebooks/          # Jupyter notebooks for interactive analysis
├── src/                # Source code for the project
│   ├── explore_data.py # Initial data exploration
│   ├── preprocess.py   # Data cleaning and feature engineering
│   ├── train.py        # Model training and selection
│   └── predict.py      # Inference and sample predictions
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd House_Price_Prediction
   ```

2. **Set up a virtual environment (recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preprocessing

Clean the raw data and prepare it for training:

```bash
python src/preprocess.py
```

### 2. Model Training

Train various regression models and save the best-performing one to the `models/` directory:

```bash
python src/train.py
```

### 3. Price Prediction

Run predictions using the trained model:

```bash
python src/predict.py
```

## Technologies Used

- **Python 3.x**
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computing.
- **Scikit-learn**: Machine learning algorithms and hyperparameter tuning.
- **Matplotlib & Seaborn**: Data visualization.
- **Joblib**: Model serialization.

## License

This project is licensed under the MIT License - see the [LICENSE](file:///x:/Works/Projects/House_Price_Prediction/LICENSE) file for details.
