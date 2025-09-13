# 🐊 Global Crocodile Species Analysis

This repository presents a **comprehensive analysis of crocodile species data** across multiple regions, habitats, and conservation statuses. The project includes a curated dataset (`crocodile_dataset.csv`) and a Python analysis script (`global_crocodile_species_analysis.py`) that performs **data exploration, visualization, preprocessing, and regression modeling** to predict crocodile lengths.

---

## 📌 Overview

- **Goal**: Predict crocodile length using regression models based on weight, age class, sex, habitat, and other features.
- **Approach**: 
  1. Perform exploratory data analysis (EDA).
  2. Visualize key distributions and relationships.
  3. Preprocess data (encoding, scaling).
  4. Train multiple regression models.
  5. Evaluate and rank models by performance.

---

## ⚙️ Requirements

- **Python**: 3.12 or higher  
- **Libraries**:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 📂 Dataset

- **File**: `crocodile_dataset.csv`  
- **Size**: 1000 observations  
- **Columns**:
  - `Observation ID`: Unique identifier  
  - `Common Name`, `Scientific Name`  
  - `Family`, `Genus`  
  - `Observed Length (m)`, `Observed Weight (kg)`  
  - `Age Class`, `Sex`  
  - `Date of Observation`, `Country/Region`, `Habitat Type`  
  - `Conservation Status` (IUCN categories)  
  - `Observer Name`, `Notes`  

⚠️ Note: The dataset is **synthetic** (randomized notes, some unrealistic values).

---

## 👀 Dataset Preview

```csv
Observation ID,Common Name,Scientific Name,Family,Genus,Observed Length (m),Observed Weight (kg),Age Class,Sex,Date of Observation,Country/Region,Habitat Type,Conservation Status,Observer Name,Notes
1,Morelet's Crocodile,Crocodylus moreletii,Crocodylidae,Crocodylus,1.9,62,Adult,Male,31-03-2018,Belize,Swamps,Least Concern,Allison Hill,Cause bill scientist nation opportunity.
2,American Crocodile,Crocodylus acutus,Crocodylidae,Crocodylus,4.09,334.5,Adult,Male,28-01-2015,Venezuela,Mangroves,Vulnerable,Brandon Hall,Ago current practice nation determine operation speak according.
3,Orinoco Crocodile,Crocodylus intermedius,Crocodylidae,Crocodylus,1.08,118.2,Juvenile,Unknown,07-12-2010,Venezuela,Flooded Savannas,Critically Endangered,Melissa Peterson,Democratic shake bill here grow gas enough analysis least by two.
...
```

---

## 📊 Analysis Workflow

### 1. Data Inspection
- Check shape, data types, missing values, and duplicates.

### 2. Exploratory Data Analysis (EDA)
- Histograms: length and weight distributions.
- Bar plots: genus, age class, sex, conservation status.
- Scatter plot: observed length vs. observed weight.

### 3. Preprocessing
- Drop irrelevant columns.
- One-hot encode categorical variables.
- Scale features and target using `StandardScaler`.

### 4. Model Training
- Algorithms:
  - **Linear Regression**
  - **Decision Tree**
  - **Random Forest**
  - **Gradient Boosting**
- Train/test split: 70/30.

### 5. Evaluation
- Metrics: **MSE, R², MAE, RMSE**
- Residual analysis
- Model ranking

---

## 📈 Sample Visualizations

- **Histograms**: Length & Weight  
- **Bar Charts**: Genus, Age Class, Sex, Conservation Status  
- **Scatter Plot**: Length vs Weight  
- **Prediction Scatter Plots**: Actual vs Predicted  
- **Residual Distributions**

---

## 🧪 Example Results (Illustrative)

```text
Linear Regressor
============================================================
MSE: 0.1234
R²: 0.8765
MAE: 0.2345
RMSE: 0.3512

Random Forest Regressor
============================================================
MSE: 0.0567
R²: 0.9432
MAE: 0.1789
RMSE: 0.2381
```

**Model Rankings**  
- By **MSE**:  
  1. Random Forest  
  2. Gradient Boosting  
  3. Linear Regression  
  4. Decision Tree  

- By **R²**:  
  1. Random Forest  
  2. Gradient Boosting  
  3. Linear Regression  
  4. Decision Tree  

---

## ▶️ How to Run

1. Ensure the following files are in the same directory:  
   - `crocodile_dataset.csv`  
   - `global_crocodile_species_analysis.py`

2. Run the script:

```bash
python global_crocodile_species_analysis.py
```

3. The script outputs:
   - Data overview
   - Visualizations
   - Model performance metrics
   - Ranked results

---

## 📌 Notes

- Dataset contains **synthetic values** (demonstration only).
- **Target variable**: `Observed Length (m)`.  
- Predictions use **weight and encoded categorical features**.  
- Results can be improved via:
  - Hyperparameter tuning
  - Advanced feature engineering  

---

## 🚀 Uploading to GitHub

```bash
git init
git add README.md crocodile_dataset.csv global_crocodile_species_analysis.py
git commit -m "Initial commit with dataset and analysis script"
git remote add origin https://github.com/igmoiiz/Crocodile-Species-Length-Analysis-Regression.git
git push -u origin main
```

---

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this work.

---
