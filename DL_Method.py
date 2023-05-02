import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
data = pd.read_csv("data_clean.csv")

# Convert 'Country' and 'Status' columns to integers
le_country = LabelEncoder()
le_status = LabelEncoder()

# Split data into features (X) and target (y)
X = data.drop('Life expectancy', axis=1)
y = data['Life expectancy']

# Perform Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature Selection
selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Create and train the MLP Regressor
mlp_regressor = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp_regressor.fit(X_train, y_train)

# Make predictions
y_pred = mlp_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R2 Score: {r2:.2f}')

# Save predictions to a DataFrame
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.reset_index(drop=True, inplace=True)
predictions.to_csv('predictions.csv', index=False)

# Plot histograms for the original data and predicted data
fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

axs[0].hist(y_test, bins=20)
axs[0].set_xlabel('Life Expectancy')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Original Data')

axs[1].hist(y_pred, bins=20)
axs[1].set_xlabel('Life Expectancy')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Predicted Data')

plt.tight_layout()
plt.savefig('images/histograms.png')
plt.show()
