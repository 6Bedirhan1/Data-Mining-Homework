# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Set style for seaborn visualizations
sns.set(style="whitegrid")

# Step 1: Load Dataset
data = load_iris()  # Load Iris dataset
X = pd.DataFrame(data.data, columns=data.feature_names)  # Create DataFrame from features
y_true = data.target  # Extract true target labels
print("Data loaded successfully. Features are:")
print(X.head())  # Display first 5 rows of data

# Step 2: Data Preprocessing
# Standardize features to ensure uniform range
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Apply scaling to features
print("Features scaled successfully.")

# Step 3: Feature Selection
# Example: Remove a feature based on domain knowledge or statistical correlation (here just an example)
# If you want to drop a feature (column), you could uncomment this:
# X_scaled = X_scaled[:, [0, 1, 2]]  # Only take first 3 features for simplicity
print("Features selected successfully.")

# Step 4: Hyperparameter Tuning - Determine Optimal Number of Clusters
inertia = []
silhouette_scores = []

# Experiment with number of clusters from 1 to 10
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Train KMeans with k clusters
    kmeans.fit(X_scaled)  # Fit the model
    inertia.append(kmeans.inertia_)  # Store inertia
    # Calculate silhouette only for k > 1
    if k > 1:
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(np.nan)

# Plot Elbow Method and Silhouette Score Graphs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method Analysis')

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores[1:], marker='o', color='orange')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.tight_layout()
plt.show()

# Select optimal k using visualizations (e.g., k=3 is a reasonable choice here)
optimal_k = 3
print(f"Optimal number of clusters determined as k={optimal_k}")

# Step 5: Fit KMeans with optimal k value
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X_scaled)
labels = kmeans.labels_  # Cluster assignments for each data point

# Visualize the clustering using PCA to reduce features to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
for i in range(optimal_k):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i+1}')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('KMeans Clustering with PCA Visualization')
plt.legend()
plt.show()

# Evaluate Silhouette Score with optimal clusters
silhouette_avg = silhouette_score(X_scaled, labels)
print(f"Silhouette Score for optimal clusters (k={optimal_k}): {silhouette_avg:.2f}")