import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data_clean.csv')

# Split data into features (X) and target (y)
X = data.drop('Life expectancy', axis=1)
y = data['Life expectancy']

# Apply feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply feature selection
kbest = SelectKBest(f_regression, k=10)  # Select top 10 features
X_selected = kbest.fit_transform(X_scaled, y)

# Perform K-Means clustering
optimal_k = 3  # Choose an appropriate number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_selected)

# Train a regression model on each cluster and make predictions
y_preds = []

for cluster_idx in range(optimal_k):
    # Get the training data for the current cluster
    X_cluster = X_selected[kmeans.labels_ == cluster_idx]
    y_cluster = y.iloc[kmeans.labels_ == cluster_idx]

    # Train a regression model on the current cluster
    regression = LinearRegression()
    regression.fit(X_cluster, y_cluster)

    # Make predictions for the current cluster
    y_preds_cluster = regression.predict(X_cluster)
    y_preds.append(y_preds_cluster)

# Combine predictions from all clusters
y_pred = np.concatenate(y_preds)

# Scatter plot for actual vs. predicted life expectancy
plt.scatter(y, y_pred)
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.title('Actual vs. Predicted Life Expectancy')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red')  # Add a diagonal line
plt.show()

# Save the predictions to a CSV file
predictions = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
predictions.to_csv('ml_predictions.csv', index=False)

# Plot histograms for the actual and predicted life expectancy
plt.hist(y, bins=20, alpha=0.5, label='Actual')
plt.hist(y_pred, bins=20, alpha=0.5, label='Predicted')
plt.xlabel('Life Expectancy')
plt.ylabel('Frequency')
plt.title('Life Expectancy Histogram')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
print(f'RMSE: {rmse}')
print(f'R-squared: {r2}')
