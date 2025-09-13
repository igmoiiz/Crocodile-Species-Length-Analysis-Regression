Global Crocodile Species Analysis
Overview
This repository contains a dataset of crocodile observations across various species, locations, and conservation statuses. The accompanying Python script performs exploratory data analysis, visualizations, data preprocessing, and applies machine learning regression models to predict the observed length of crocodiles based on features like weight, age class, sex, habitat, and more.
The analysis aims to understand distributions, relationships in the data, and evaluate model performances for length prediction.
Requirements

Python 3.12 or higher
Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn (for preprocessing, model training, and metrics)

Install the required libraries using:
pip install pandas numpy matplotlib seaborn scikit-learn

Dataset
The dataset is provided in crocodile_dataset.csv. It includes 1000 observations with the following columns:

Observation ID: Unique identifier for each observation.
Common Name: Common name of the crocodile species.
Scientific Name: Scientific name of the species.
Family: Taxonomic family (e.g., Crocodylidae).
Genus: Taxonomic genus (e.g., Crocodylus, Osteolaemus).
Observed Length (m): Length of the observed crocodile in meters.
Observed Weight (kg): Weight of the observed crocodile in kilograms.
Age Class: Age category (e.g., Adult, Juvenile, Subadult, Hatchling).
Sex: Sex of the crocodile (Male, Female, Unknown).
Date of Observation: Date when the observation was made (DD-MM-YYYY).
Country/Region: Location of the observation.
Habitat Type: Type of habitat (e.g., Rivers, Swamps, Mangroves).
Conservation Status: IUCN conservation status (e.g., Least Concern, Vulnerable, Critically Endangered).
Observer Name: Name of the observer.
Notes: Additional notes (random text in this dataset).

Dataset Preview
Observation ID,Common Name,Scientific Name,Family,Genus,Observed Length (m),Observed Weight (kg),Age Class,Sex,Date of Observation,Country/Region,Habitat Type,Conservation Status,Observer Name,Notes
1,Morelet's Crocodile,Crocodylus moreletii,Crocodylidae,Crocodylus,1.9,62,Adult,Male,31-03-2018,Belize,Swamps,Least Concern,Allison Hill,Cause bill scientist nation opportunity.
2,American Crocodile,Crocodylus acutus,Crocodylidae,Crocodylus,4.09,334.5,Adult,Male,28-01-2015,Venezuela,Mangroves,Vulnerable,Brandon Hall,Ago current practice nation determine operation speak according.
3,Orinoco Crocodile,Crocodylus intermedius,Crocodylidae,Crocodylus,1.08,118.2,Juvenile,Unknown,07-12-2010,Venezuela,Flooded Savannas,Critically Endangered,Melissa Peterson,Democratic shake bill here grow gas enough analysis least by two.
4,Morelet's Crocodile,Crocodylus moreletii,Crocodylidae,Crocodylus,2.42,90.4,Adult,Male,01-11-2019,Mexico,Rivers,Least Concern,Edward Fuller,Officer relate animal direction eye bag do.
5,Mugger Crocodile (Marsh Crocodile),Crocodylus palustris,Crocodylidae,Crocodylus,3.75,269.4,Adult,Unknown,15-07-2019,India,Rivers,Vulnerable,Donald Reid,Class great prove reduce raise author play move each left establish understand read detail.
... (Full dataset contains 1000 rows; see crocodile_dataset.csv for complete data)

Note: The dataset appears to be synthetic or sample data, with notes containing random text. Some values may be unrealistic (e.g., hatchlings with high weights), as it's for demonstration purposes.
Analysis Script
The Python script global_crocodile_species_analysis.py loads the dataset, performs EDA (exploratory data analysis), preprocesses the data, trains regression models, evaluates them, and ranks the models based on performance metrics.
Key Steps in the Script:

Load and Inspect Data: Read CSV, check shape, dtypes, missing values, duplicates.
Visualizations: Histograms for length and weight, bar plots for categorical features (genus, age class, sex, conservation status), scatter plot for length vs. weight.
Data Preprocessing: Drop unnecessary columns, encode categorical variables using one-hot encoding, scale features and target.
Model Training: Split data into train/test, train Linear Regression, Decision Tree, Random Forest, and Gradient Boosting regressors.
Evaluation: Calculate MSE, R2, MAE, RMSE for each model; visualize predictions and residuals.
Model Ranking: Rank models by MSE (lower better) and R2 (higher better).

Full Script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.simplefilter('ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 5)
sns.set(color_codes=True)

# Importing the dataset
df = pd.read_csv('crocodile_dataset.csv')
df.head(5)

# Checking the Dataset
df.shape
df.dtypes
df.info()

# Drawing a heatmap to check for null values
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='tab20c_r')
plt.title("Missing Data: Training Set")
plt.show()

# Printing Duplicated and Null Values
print("Duplicated Values: ", df.duplicated().sum())
print("Null Values: ", df.isnull().sum())

# Distribution of Observed Length
df['Observed Length (m)'].plot(kind='hist', bins=20, title='Observed Length (m)')
plt.gca().spines[['top', 'right']].set_visible(False)

# Distribution of Observed Weight
df['Observed Weight (kg)'].plot(kind='hist', bins=20, title='Observed Weight (kg)')
plt.gca().spines[['top', 'right']].set_visible(False)

# Distribution of Genus of Crocodile Species
df.groupby('Genus').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.title("Distribution Relationship of Genus")
plt.gca().spines[['top', 'right']].set_visible(False)

# Distribution of Age Class of Crocodile Species
df.groupby('Age Class').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.title("Distribution Relationship of Age Class")
plt.gca().spines[['top', 'right']].set_visible(False)

# Distribution of Sex of Crocodile Species
df.groupby('Sex').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.title("Distribution Relationship of Sex")
plt.gca().spines[['top', 'right']].set_visible(False)

# Distribution of Conservation Status of Crocodile Species
df.groupby('Conservation Status').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.title("Distribution Relationship of Conservation Status")
plt.gca().spines[['top', 'right']].set_visible(False)

# Scatterplot to show the relationship between observed length and weight
df.plot(kind='scatter', x='Observed Length (m)', y='Observed Weight (kg)', s=32, alpha=.8)
plt.title("Observed Length vs Observed Weight")
plt.gca().spines[['top', 'right']].set_visible(False)

# Dropping Unnecessary columns
df.drop(['Observation ID', 'Family', 'Genus', 'Date of Observation', 'Observer Name', 'Notes'], axis=1, inplace=True)

df.drop('Common Name', axis=1, inplace=True)
df.drop('Country/Region', axis=1, inplace=True)

# Preparing dummy data for encoding categorical variables
name = pd.get_dummies(df['Scientific Name'], drop_first=True)
age_class = pd.get_dummies(df['Age Class'], drop_first=True)
sex = pd.get_dummies(df['Sex'], drop_first=True)
conservation_status = pd.get_dummies(df['Conservation Status'], drop_first=True)
habitat_type = pd.get_dummies(df['Habitat Type'], drop_first=True)

# Dropping original variables
df.drop(['Scientific Name', 'Age Class', 'Sex', 'Conservation Status', 'Habitat Type'], axis=1, inplace=True)

# Replacing with dummy variables
df = pd.concat([df, name, age_class, sex, conservation_status, habitat_type], axis=1)

# Selecting all the columns with boolean data types
bool_columns = df.select_dtypes(include='bool').columns

# Type converting all the boolean variable types into integer in the whole dataset
for colname in bool_columns:
    df[bool_columns] = df[bool_columns].astype(int)

# Selecting our Target Variable and Features
x = df.drop('Observed Length (m)', axis=1)
y = df['Observed Length (m)']

# Preprocessing the Input Feature Variables
pre_processor_x = preprocessing.StandardScaler().fit(x)
x_transform = pre_processor_x.fit_transform(x)

# Preprocessing the Output Target Variable
pre_processor_y = preprocessing.StandardScaler().fit(y.values.reshape(-1, 1))
y_transform = pre_processor_y.fit_transform(y.values.reshape(-1, 1))

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_transform, y_transform, test_size=0.30, random_state=98)

# Importing Regression Models
linear_model = LinearRegression()
tree_model = DecisionTreeRegressor()
forest_model = RandomForestRegressor()
gradient_model = GradientBoostingRegressor()

# Fitting the models
linear_model.fit(x_train, y_train)
tree_model.fit(x_train, y_train)
forest_model.fit(x_train, y_train)
gradient_model.fit(x_train, y_train)

# Getting Predictions from the model
y_pred_linear = linear_model.predict(x_test)
y_pred_tree = tree_model.predict(x_test)
y_pred_forest = forest_model.predict(x_test)
y_pred_gradient = gradient_model.predict(x_test)

# ScatterPlot for visualizing the data after transformation
for y_pred in [y_pred_linear, y_pred_tree, y_pred_forest, y_pred_gradient]:
    y_test_1d = y_test.flatten() if y_test.ndim > 1 else y_test
    y_pred_1d = y_pred.flatten() if y_pred.ndim > 1 else y_pred
    sns.scatterplot(x=y_test_1d, y=y_pred_1d, color='blue', label='Actual Data Points')
    plt.plot([min(y_test_1d), max(y_test_1d)], [min(y_test_1d), max(y_test_1d)], color='red', label="Ideal Line")
    plt.legend()
    plt.show()

# Metrics Calculation
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_tree = mean_squared_error(y_test, y_pred_tree)
mse_forest = mean_squared_error(y_test, y_pred_forest)
mse_gradient = mean_squared_error(y_test, y_pred_gradient)

r2_linear = r2_score(y_test, y_pred_linear)
r2_tree = r2_score(y_test, y_pred_tree)
r2_forest = r2_score(y_test, y_pred_forest)
r2_gradient = r2_score(y_test, y_pred_gradient)

mae_linear = mean_absolute_error(y_test, y_pred_linear)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
mae_forest = mean_absolute_error(y_test, y_pred_forest)
mae_gradient = mean_absolute_error(y_test, y_pred_gradient)

rmse_linear = np.sqrt(mse_linear)
rmse_tree = np.sqrt(mse_tree)
rmse_forest = np.sqrt(mse_forest)
rmse_gradient = np.sqrt(mse_gradient)

# Printing the results
print("Linear Regressor")
print("==" * 30)
print(f"MSE: {mse_linear}")
print(f"R2: {r2_linear}")
print(f"MAE: {mae_linear}")
print(f"RMSE: {rmse_linear}")
print("==" * 30)

print("\nDecision Tree Regressor")
print("==" * 30)
print(f"MSE: {mse_tree}")
print(f"R2: {r2_tree}")
print(f"MAE: {mae_tree}")
print(f"RMSE: {rmse_tree}")
print("==" * 30)

print("\nRandom Forest Regressor")
print("==" * 30)
print(f"MSE: {mse_forest}")
print(f"R2: {r2_forest}")
print(f"MAE: {mae_forest}")
print(f"RMSE: {rmse_forest}")
print("==" * 30)

print("\nGradient Boosting Regressor")
print("==" * 30)
print(f"MSE: {mse_gradient}")
print(f"R2: {r2_gradient}")
print(f"MAE: {mae_gradient}")
print(f"RMSE: {rmse_gradient}")
print("==" * 30)

# Distribution plot to check residuals
for y_pred in [y_pred_linear, y_pred_tree, y_pred_forest, y_pred_gradient]:
    residual = y_test - y_pred
    sns.distplot(residual, kde=True)
    plt.title("Distribution Plot for Residual Values")
    plt.show()

# Comparison of results according to mean square error
model_mse_scores = {
    "Linear Regression": mse_linear,
    "Decision Tree": mse_tree,
    "Random Forest": mse_forest,
    "Gradient Boosting Regressor": mse_gradient
}

sorted_scores = sorted(model_mse_scores.items(), key=lambda x: x[1])
print("Model Ranking - (Lower Values are better)")
for rank, (model_name, score) in enumerate(sorted_scores, start=1):
    print(f"{rank}. {model_name}: {score}")

# Comparison of results according to r2 value
model_r2_scores = {
    "Linear Regression": r2_linear,
    "Decision Tree": r2_tree,
    "Random Forest": r2_forest,
    "Gradient Boosting Regressor": r2_gradient
}

sorted_scores = sorted(model_r2_scores.items(), key=lambda x: x[1], reverse=True)
print("\nModel Ranking - (Higher Values are better)")
for rank, (model_name, score) in enumerate(sorted_scores, start=1):
    print(f"{rank}. {model_name}: {score}")

How to Run

Ensure crocodile_dataset.csv and global_crocodile_species_analysis.py are in the same directory.
Install dependencies:pip install pandas numpy matplotlib seaborn scikit-learn


Run the script:python global_crocodile_species_analysis.py


The script will display data info, visualizations, metrics, and model rankings.

Sample Output
Note: Actual values depend on the dataset. Example output (values are illustrative):
Duplicated Values: 0
Null Values: Observation ID         0
Common Name           0
Scientific Name       0
...

Linear Regressor
============================================================
MSE: 0.1234
R2: 0.8765
MAE: 0.2345
RMSE: 0.3512
============================================================

Decision Tree Regressor
============================================================
MSE: 0.1345
R2: 0.8654
MAE: 0.2456
RMSE: 0.3667
============================================================

Random Forest Regressor
============================================================
MSE: 0.0567
R2: 0.9432
MAE: 0.1789
RMSE: 0.2381
============================================================

Gradient Boosting Regressor
============================================================
MSE: 0.0678
R2: 0.9321
MAE: 0.1923
RMSE: 0.2604
============================================================

Model Ranking - (Lower Values are better)
1. Random Forest: 0.0567
2. Gradient Boosting Regressor: 0.0678
3. Linear Regression: 0.1234
4. Decision Tree: 0.1345

Model Ranking - (Higher Values are better)
1. Random Forest: 0.9432
2. Gradient Boosting Regressor: 0.9321
3. Linear Regression: 0.8765
4. Decision Tree: 0.8654

Visualizations

Histograms: Distribution of observed length and weight.
Bar Plots: Distribution of genus, age class, sex, and conservation status.
Scatter Plot: Observed length vs. weight.
Prediction Scatter Plots: Predicted vs. actual length for each model.
Residual Plots: Distribution of residuals for each model.

To save visualizations, modify the script to include plt.savefig('filename.png') after each plt.show().
Notes

The dataset has no missing values or duplicates.
The target variable is Observed Length (m), predicted using weight and encoded categorical features.
Models could be improved with hyperparameter tuning or feature engineering.
The Notes column contains random text and is dropped during preprocessing.
Some data points (e.g., hatchling weights) may be unrealistic due to the synthetic nature of the dataset.

Uploading to GitHub

Save this file as README.md.
Create a GitHub repository.
Upload README.md, crocodile_dataset.csv, and global_crocodile_species_analysis.py to the repository.
Use the following steps to upload:git init
git add README.md crocodile_dataset.csv global_crocodile_species_analysis.py
git commit -m "Initial commit with dataset and analysis script"
git remote add origin <https://github.com/igmoiiz/Crocodile-Species-Length-Analysis-Regression.git>
git push -u origin main



License
This project is licensed under the MIT License. Feel free to use, modify, and distribute.