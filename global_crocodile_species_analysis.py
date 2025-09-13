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
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10, 5)
sns.set(color_codes=True)

# Improting the dataset
df = pd.read_csv('crocodile_dataset.csv')
df.head(5)

# Checking the Dataset
df.shape

df.dtypes

df.info()

# Drawing a heatmap to check for null values
sns.heatmap(df.isnull(), yticklabels = False, cbar=False, cmap='tab20c_r')
plt.title("Missing Data: Training Set")
plt.show()

# Printing Duplicated and Null Values
print("Duplicated Values: ", df.duplicated().sum())
print("Null Values: ", df.isnull().sum())

df.head(5)

# Distribution of Observed Length
df['Observed Length (m)'].plot(kind='hist', bins=20, title='Observed Length (m)')
plt.gca().spines[['top', 'right',]].set_visible(False)

# Distribution of Observed Weight
df['Observed Weight (kg)'].plot(kind='hist', bins=20, title='Observed Weight (kg)')
plt.gca().spines[['top', 'right',]].set_visible(False)

# Distribution of Genus of Crocodile Species
df.groupby('Genus').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.title("Distribution Relationship of Genus")
plt.gca().spines[['top', 'right',]].set_visible(False)

# Distribution of age class of Crocodile Species
df.groupby('Age Class').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.title("Dustribution Relationship of Age Class")
plt.gca().spines[['top', 'right',]].set_visible(False)

# Distribution of Sex of Crocodile Species
df.groupby('Sex').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.title("Dustribution Relationship of Sex")
plt.gca().spines[['top', 'right',]].set_visible(False)

# Distribution of Conservation Status of Crocodile Species espacially the endangered ones
df.groupby('Conservation Status').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.title("Dustribution Relationship of Conservation Status")
plt.gca().spines[['top', 'right',]].set_visible(False)

# Scatterplot to show the relationship between observed length and weight
df.plot(kind='scatter', x='Observed Length (m)', y='Observed Weight (kg)', s=32, alpha=.8)
plt.title("Observed Length vs Observed Weight")
plt.gca().spines[['top', 'right',]].set_visible(False)

# Dropping Unnecessary columns
df.drop(['Observation ID', 'Family', 'Genus', 'Date of Observation', 'Observer Name', 'Notes'], axis=1, inplace=True)
df.head(5)

df.shape

df.drop('Common Name', axis=1, inplace=True)
df.head(5)

df.drop('Country/Region', axis=1, inplace = True)
df.head(5)

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

df.head(5)

df.shape

# Selecting all the columns with boolean data types
bool_columns = df.select_dtypes(include='bool').columns

# Type converting all the boolean variable types into integer in the whole dataset
for colname in bool_columns:
  df[bool_columns] = df[bool_columns].astype(int)

df.head(5)

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
  # Ensure y_test and y_pred are 1-dimensional
  y_test_1d = y_test.flatten() if y_test.ndim > 1 else y_test
  y_pred_1d = y_pred.flatten() if y_pred.ndim > 1 else y_pred
  sns.scatterplot(x=y_test_1d, y=y_pred_1d, color='blue', label='Actual Data Points')
  plt.plot([min(y_test_1d), max(y_test_1d)], [min(y_test_1d), max(y_test_1d)], color='red', label="Ideal Line")
  plt.legend()
  plt.show()

# Metrics Calculation

#========Mean Square Error=================#
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_tree = mean_squared_error(y_test, y_pred_tree)
mse_forest = mean_squared_error(y_test, y_pred_forest)
mse_gradient = mean_squared_error(y_test, y_pred_gradient)

#========R2 Value=================#
r2_linear = r2_score(y_test, y_pred_linear)
r2_tree = r2_score(y_test, y_pred_tree)
r2_forest = r2_score(y_test, y_pred_forest)
r2_gradient = r2_score(y_test, y_pred_gradient)

#========Mean Absolute Error=================#
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
mae_forest = mean_absolute_error(y_test, y_pred_forest)
mae_gradient = mean_absolute_error(y_test, y_pred_gradient)

#========Root Mean Square Error=================#
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

print("\n")

print("Decision Tree Regressor")
print("==" * 30)
print(f"MSE: {mse_tree}")
print(f"R2: {r2_tree}")
print(f"MAE: {mae_tree}")
print(f"RMSE: {rmse_tree}")
print("==" * 30)

print("\n")

print("Random Forest Regressor")
print("==" * 30)
print(f"MSE: {mse_forest}")
print(f"R2: {r2_forest}")
print(f"MAE: {mae_forest}")
print(f"RMSE: {rmse_forest}")
print("==" * 30)

print("\n")

print("Gradient Boosting Regressor")
print("==" * 30)
print(f"MSE: {mse_gradient}")
print(f"R2: {r2_gradient}")
print(f"MAE: {mae_gradient}")
print(f"RMSE: {rmse_gradient}")
print("==" * 30)

# Distribution plot to check if the result is somewhat different from the non transformed data or not
for y_pred in [y_pred_linear, y_pred_tree, y_pred_forest, y_pred_gradient]:
  residual = y_test - y_pred
  sns.distplot(residual, kde=True)
  plt.title("Distribution Plot for Residual Values")
  plt.show()

# Comparison of results according to mean square error
model_mse_scores = {
    "Linear Regression" : mse_linear,
    "Decision Tree" : mse_tree,
    "Random Forest" : mse_forest,
    "Gradient Boosting Regressor" : mse_gradient
}

# Sort the model scores in ascending order based on their values (lower values first)
sorted_scores = sorted(model_mse_scores.items(), key=lambda x: x[1])

# Display the ranking of models
print("Model Ranking - (Lower Values are better)")
for rank, (model_name, score) in enumerate(sorted_scores, start=1):
    print(f"{rank}. {model_name}: {score}")

# Comparison of results according to r2 value
model_r2_scores = {
    "Linear Regression" : r2_linear,
    "Decision Tree" : r2_tree,
    "Random Forest" : r2_forest,
    "Gradient Boosting Regressor" : r2_gradient
}

# Sort the model scores in ascending order based on their values (lower values first)
sorted_scores = sorted(model_r2_scores.items(), key=lambda x: x[1])

# Display the ranking of models
print("Model Ranking - (Higher Values are better)")
for rank, (model_name, score) in enumerate(sorted_scores, start=1):
    print(f"{rank}. {model_name}: {score}")