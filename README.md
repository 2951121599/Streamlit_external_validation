# External Validation of Prediction Model

This Streamlit application performs external validation of the prediction model using an independent dataset. The application:

- Loads and processes the external validation dataset
- Applies the trained model for predictions
- Generates ROC curves to visualize model performance
- Calculates key evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

## Setup and Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:

```bash
streamlit run app.py
```
