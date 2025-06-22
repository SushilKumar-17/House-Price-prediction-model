
# Assignment 5: Bengaluru House Price Prediction 

A complete machine learning pipeline that predicts house prices in Bengaluru based on factors like size, location, square footage, number of bathrooms, etc. This project includes data cleaning, feature engineering, exploratory analysis, model training, and a Streamlit deployment.
<div align="center">
  <img src="assets/img (1).png" alt="App Demo" width="800"/>
---
  <img src="assets/img (2).png" alt="App Demo" width="800"/>
</div>

---
## 📂 Folder Structure

├── Assignment 5.ipynb # Jupyter Notebook with full analysis and modeling<br>
├── app.py # Streamlit App for model prediction<br>
├── Bengaluru_House_Data.csv # Original dataset<br>
├── final_df.csv # Final cleaned & selected features<br>
├── houseF_encoded.csv # One-hot encoded version (with location)<br>
├── house_price_model.pkl # Trained model (pickle format)<br>
├── model_columns.pkl # Column names used during training<br>
├── .venv/ # Virtual environment (optional)<br>
---

## Exploratory Data Analysis (EDA)

- **Univariate Analysis**: Visualized distributions for `price`, `total_sqft`, `bath`, and `bhk`.
- **Bivariate Analysis**: Explored relationships like `price vs total_sqft`, `price vs bhk`, and `price per sqft by location`.

### Key Findings:
- Prices showed right-skewed distribution.
- Outliers existed in sqft/price, especially in locations with low sample size.
- Strong correlation between `total_sqft`, `bath`, `bhk` and price.

---

## Feature Engineering

### ✅ Derived Features:
- **`bhk`**: Extracted from `size` column (e.g., "2 BHK" → 2).
- **`total_sqft`**: Cleaned non-numeric values like `'2100 - 2850'` to mean values.
- **`price_per_sqft`**: Engineered for location-based normalization and anomaly detection.
- **`location`**: Grouped rare locations into `'other'` if occurrences < 10.

### 🚫 Dropped:
- `society`, `availability`, `area_type`: Low predictive value or high missingness.
- `price_per_sqft`: Created only for outlier handling, dropped before final model.

---

## Feature Selection

Final selected features (stored in `final_df.csv`):
- `total_sqft`
- `bath`
- `bhk`
- `location` (one-hot encoded in `houseF_encoded.csv`)

### One-hot Encoding Logic:
- Performed on `location` after grouping infrequent locations into `'other'`.
- Stored in `houseF_encoded.csv`.
- Resulting model columns saved in `model_columns.pkl` for future inference.

---

## Model Training

### Trained Two Models:
1. **Model 1 (All Features)**: Included `total_sqft`, `bath`, `bhk`, one-hot encoded location.
2. **Model 2 (Selective Features)**: Removed features with lower contribution from EDA insights.

#### 🔍 Final Decision:
- **Model 1 was retained**, as it gave better cross-validation scores.
- Saved as `house_price_model.pkl`.

---

## Streamlit Web App

The trained model was deployed as a **Streamlit web application** via `app.py`.

### Features:
- User can enter `location`, `total_sqft`, `bath`, and `bhk`.
- Model predicts price instantly.
- Location drop-down dynamically populated using `model_columns.pkl`.

---

## Conclusion

This project demonstrates a complete machine learning workflow including:

- Cleaning and transforming raw real estate data
- Engineering robust features
- Choosing the right model based on validation metrics
- Building a production-ready web app for deployment

---

## Author

**Sushil Kumar Patra**  
Data Science Intern
