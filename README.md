# 🏡 XAI Dashboard — Explainable AI for House Price Prediction

In this project an interactive application designed to predict house prices while providing explainable insights into how the model makes decisions has been created with the help of streamlit for hosting.

This dashboard leverages **SHAP (SHapley Additive exPlanations)** to deliver both local and global interpretability, helping users understand which features influence the prediction and how.

> [View Live Dashboard](https://xai-dashboard-hmxekytzgzxnxejdhndptt.streamlit.app)

---

##  Features

###  Local Explanations
- **Feature Impact View**: Explore how each feature (e.g., Area, Bedrooms) affects the prediction for an individual house.
- **Interactive Scenarios**: Adjust inputs like area size via sliders and observe real-time changes in the predicted price.
- **Treemap Visualization**: View a treemap where size and color indicate the feature's contribution to price — positive (green) or negative (red).
- **Textual Insights**: Natural language summaries explain which features drive prices up or down.

### Global Explanations
- **Feature Importance Ranking**: Discover the top and bottom features globally influencing the model using SHAP value averages.
- **Bar Charts and Grids**: Visual comparison of feature impacts across the training dataset.
- **Feature Effect Plots**: See how changes in a single feature affect predicted prices via line plots.

###  Data Exploration
- **Data Distributions**: Visualize feature distributions (e.g., Area, Price) across the dataset with histograms.
- **Instance Highlighting**: Your input is marked on the distributions for comparison.
- **Feature Effect Analysis**: Select a feature and see how varying it changes predictions.

---

##  Technologies Used

- **Machine Learning**: Random Forest Regressor (scikit-learn)
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Web App Framework**: Streamlit
- **Visualization Libraries**:
  - Matplotlib
  - Seaborn
  - Squarify
- **Data Handling**:
  - Pandas
  - NumPy

---

## Project Structure

- `main.py`: Main Streamlit application.
- `house_price_rf.pkl`: Trained Random Forest model.
- `shap_explainer.pkl`: Trained SHAP explainer model.
- `train_features.csv`: Dataset used for SHAP global explanations.
- `Houses.csv`: Full dataset for user interaction and visualization.

---

## Running Locally

1. Clone this repository:
    ```bash
    git clone https://github.com/alexcega/XAI-Dashboard.git
    cd XAI-Dashboard
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch the Streamlit app:
    ```bash
    streamlit run main.py
    ```
    Or
    ```bash
    python -m streamlit run main.py
    ```

> **Note**: Make sure you have `predictor.pkl`, `explainer.pkl`, and the necessary CSV files in your working directory.

---

## Datasets

- **Train Features**: `train_features.csv` — Used for computing global SHAP value summaries.
- **Full Dataset**: Loaded for live interaction, scenario simulation, and market comparisons.

The dataset was optained from Kaggle as part of the [Housing prices dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

---

##  Live Demo

Check out the live deployed version here:  
[🔗 Live Dashboard](https://xai-dashboard-hmxekytzgzxnxejdhndptt.streamlit.app/#feature-impact-local-explanation)

