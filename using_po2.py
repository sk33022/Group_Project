import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score
# Load the dataset
data = pd.read_csv("po2_data (2).csv")

# Data Exploration
print(data.info())  # Display dataset information
print(data.describe())  # Summary statistics

# Data Visualization
sns.pairplot(data[['motor_updrs', 'total_updrs', 'age', 'sex', 'jitter(%)', 'shimmer(%)']])
plt.show()

# Correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Data Preprocessing
X = data[['age', 'sex', 'jitter(%)', 'shimmer(%)']]  # Predictor variables
y_motor = data['motor_updrs']  # Motor UPDRS as the target variable
y_total = data['total_updrs']  # Total UPDRS as the target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_motor_train, y_motor_test, y_total_train, y_total_test = train_test_split(
    X, y_motor, y_total, test_size=0.2, random_state=42)

# Linear Regression Modeling for Motor UPDRS
model_motor = LinearRegression()
model_motor.fit(X_train, y_motor_train)

# Linear Regression Modeling for Total UPDRS
model_total = LinearRegression()
model_total.fit(X_train, y_total_train)

# Model Evaluation for Motor UPDRS
motor_predictions = model_motor.predict(X_test)
mse_motor = mean_squared_error(y_motor_test, motor_predictions)
r2_motor = r2_score(y_motor_test, motor_predictions)
print(f'Motor UPDRS Mean Squared Error: {mse_motor}')
print(f'Motor UPDRS R-squared: {r2_motor}')

# Model Evaluation for Total UPDRS
total_predictions = model_total.predict(X_test)
mse_total = mean_squared_error(y_total_test, total_predictions)
r2_total = r2_score(y_total_test, total_predictions)
print(f'Total UPDRS Mean Squared Error: {mse_total}')
print(f'Total UPDRS R-squared: {r2_total}')

# Plotting Actual vs. Predicted for Motor UPDRS
plt.scatter(y_motor_test, motor_predictions)
plt.xlabel('Actual Motor UPDRS')
plt.ylabel('Predicted Motor UPDRS')
plt.title('Actual vs. Predicted Motor UPDRS')
plt.show()
# Assuming you have already trained the models (model_motor and model_total)
motor_predictions = model_motor.predict(X_test)
total_predictions = model_total.predict(X_test)

# Calculate R-squared for Motor UPDRS
r2_motor = r2_score(y_motor_test, motor_predictions)
print(f'Motor UPDRS R-squared: {r2_motor}')

# Calculate R-squared for Total UPDRS
r2_total = r2_score(y_total_test, total_predictions)
print(f'Total UPDRS R-squared: {r2_total}')
# Plotting Actual vs. Predicted for Total UPDRS
plt.scatter(y_total_test, total_predictions)
plt.xlabel('Actual Total UPDRS')
plt.ylabel('Predicted Total UPDRS')
plt.title('Actual vs. Predicted Total UPDRS')
plt.show()
