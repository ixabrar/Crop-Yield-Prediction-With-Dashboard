# Crop Yield Prediction System

A machine learning system for predicting crop yield in Indian states based on agricultural and environmental factors.

## Overview

This project analyzes historical crop production data and builds predictive models to forecast crop yields across different Indian states. The system includes both regression models for yield prediction and classification models for understanding production patterns.

## Features

- Comprehensive crop yield prediction using multiple ML algorithms
- Support for various crops across Indian states
- Analysis of factors including rainfall, fertilizer usage, and pesticide application
- Interactive web interface powered by Streamlit
- Multiple model types including regression and classification

## Project Structure

```
crop_yield_prediction/
├── app.py                    # Streamlit web application
├── modeltrain.py            # Model training and evaluation
├── verifyresult.py          # Result verification and testing
├── crop_yield_pred.ipynb    # Jupyter notebook for exploration
└── crop-yield-in-indian-states-dataset/
    └── crop_yield.csv       # Dataset with crop production data
```

## Dataset

The dataset contains historical crop production information for Indian states including:

- Crop type
- Year of production
- Season (Kharif, Rabi, Whole Year)
- State location
- Area under cultivation
- Production volume
- Annual rainfall
- Fertilizer usage
- Pesticide usage
- Yield per unit area

## Technologies Used

- Python 3.x
- Scikit-learn - Machine learning algorithms
- Pandas - Data manipulation and analysis
- NumPy - Numerical computing
- Streamlit - Web application framework
- Plotly - Interactive visualization
- Joblib - Model serialization

## Installation

1. Create a virtual environment:
```bash
python -m venv crop
```

2. Activate the virtual environment:
```bash
# On Windows
crop\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Web Application

```bash
streamlit run app.py
```

The application will open in your default browser at http://localhost:8501

### Train the Model

```bash
python modeltrain.py
```

This will train and evaluate the prediction models using the dataset.

### Verify Results

```bash
python verifyresult.py
```

This script validates the model predictions against test data.

## Model Approaches

The system implements multiple machine learning algorithms:

- Linear Regression for continuous yield prediction
- Decision Trees for interpretable patterns
- Random Forests for robust predictions
- Logistic Regression for classification tasks
- K-Nearest Neighbors for similarity-based predictions
- Support Vector Machines for complex pattern recognition
- Naive Bayes for probabilistic classification

## Results

The trained models are serialized using joblib and loaded in the web application for real-time predictions and interactive analysis.

## Contributing

Feel free to fork this project and submit pull requests with improvements or new features.

## License

This project is open source and available under the MIT License.

## Contact

For questions or feedback about this project, please open an issue in the repository.
