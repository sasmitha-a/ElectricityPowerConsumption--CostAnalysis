import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Read the dataset
data = pd.read_csv("proj.csv")

print("Data Head:")
print(data.head())

# Get the shape of the DataFrame (number of rows, number of columns)
print("Data Shape:")
print(data.shape)

# Get the column names of the DataFrame
print("Column Names:")
print(data.columns)

# Get the data types of each column
print("Data Types:")
print(data.dtypes)

# Get the summary statistics of the numerical columns
print("Summary Statistics:")
print(data.describe())

# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

# Select relevant columns for analysis
selected_columns = ["CurrentCharges", "Consumption(KWH)", "KWHCharges", "Consumption(KW)",
                    "KWCharges", "Othercharges"]
data = data[selected_columns]
print(data.dtypes)

# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform PCA
pca = PCA()
pca.fit(data_scaled)

feature_names = data.columns
principal_component_names = pca.get_feature_names_out(feature_names)

# Print the principal component names

# Calculate the principal components and their corresponding explained variances
principal_components = pca.components_
explained_variances = pca.explained_variance_ratio_

# Determine the number of principal components to retain
num_components = [i for i, num in enumerate(explained_variances) if num > 0.09]

reduced_column = [data.columns[i] for i in range(len(data.columns)) if i in num_components]
data_reduced = data[reduced_column]
print(data_reduced.head())

#TIMESERIES

kmeans = KMeans(n_clusters=3, n_init=10)  # Set the value of n_init explicitly

# Step 2: Load the data into a pandas DataFrame
data = pd.read_csv("C:\\Zraswanth\\4.DCS SEM-4\\LAB PA\\proj\\project.csv")

# Select relevant columns for analysis
selected_columns2 = [
    "Development Name", "Borough", "Account Name", "Location", "Meter AMR", "Meter Scope", "TDS #",
    "EDP", "RC Code", "Funding Source", "AMP #", "Vendor Name", "UMIS BILL ID", "Revenue Month",
    "Service Start Date", "Service End Date", "# days", "MeterNumber", "Estimated", "CurrentCharges",
    "Rate Class", "Bill Analyzed", "Consumption(KWH)", "KWHCharges", "Consumption(KW)",
    "KWCharges", "Othercharges"
]

selected_data2 = data[selected_columns2]

# Step 3: Convert the 'Revenue Month' column to a datetime type
data['Revenue Month'] = pd.to_datetime(data['Revenue Month'])

# Step 4: Split the data into separate columns for 'Year' and 'Month'
data['Year'] = data['Revenue Month'].dt.year
data['Month'] = data['Revenue Month'].dt.month

# Step 5: Group the data by year and month, and calculate the average 'Consumption(KW)'
grouped_data = data.groupby(['Year', 'Month']).mean()['Consumption(KW)']

# Step 6: Plot the average consumption over time
grouped_data.plot()
plt.xlabel('Year and Month')
plt.ylabel('Average Consumption(KW)')
plt.title('Average Consumption(KW) over Time')
plt.show()

# Step 7: Perform time series forecasting using ARIMA
model = ARIMA(grouped_data.values, order=(1, 0, 0))  # Adjust the order as per your data
model_fit = model.fit()
forecast = model_fit.forecast(steps=36)  # Forecasting 36 months ahead

# Step 8: Plot the forecasted values
plt.plot(grouped_data.reset_index().index, grouped_data.values, label='Actual')
plt.plot(range(len(grouped_data), len(grouped_data) + len(forecast)), forecast, label='Forecast')
plt.xlabel('Index')
plt.ylabel('Consumption(KW)')
plt.title('Actual vs Forecasted Consumption(KW)')
plt.legend()
plt.show()


# Clustering
# Perform clustering on selected features
features_for_clustering = ["Consumption(KWH)", "KWHCharges", "Consumption(KW)", "KWCharges"]
clustering_data = selected_data2[features_for_clustering].copy()

# Standardize the data
scaler = StandardScaler()
standardized_data = scaler.fit_transform(clustering_data)

# Determine the optimal number of clusters using the elbow method
inertia = []
k_values = range(1, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=123)
    kmeans.fit(standardized_data)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(k_values, inertia, marker="o")
plt.title("Elbow Curve for Optimal Cluster Selection")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# Based on the elbow curve, select the optimal number of clusters
optimal_k = 3

# Perform clustering with the selected number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=123)
cluster_labels = kmeans.fit_predict(standardized_data)

# Plot scatter plot with cluster labels
plt.scatter(data['Consumption(KWH)'],data["KWHCharges"] , c=cluster_labels, cmap='viridis')
plt.xlabel('KWCharges')
plt.ylabel('KWHCharges')
plt.title('Scatter Plot with Clusters')
plt.show()


# Add cluster labels to the selected data
selected_data2["Cluster"] = cluster_labels

# Discriminant Analysis
# Assuming 'Bill Analyzed' is the target variable for classification
target_variable = "Bill Analyzed"

# Separate the features and target variable
X = selected_data2[features_for_clustering].copy()
y = selected_data2[target_variable].copy()
