import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generate synthetic data
np.random.seed(42)
dataSize = 1000
x = np.random.rand(dataSize)
y = 2 * x + 1 + 0.1 * np.random.randn(dataSize)

# Reshape data for Keras input
data = np.column_stack((x, y))

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Build K-Means model with varying clusters
clusters_range = range(2, 10)
inertia_values = []

for n_clusters in clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)

# Plot the Elbow Method
plt.plot(clusters_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (within-cluster sum of squares)')
plt.show()

# Determine the optimal number of clusters using silhouette analysis
silhouette_scores = []

for n_clusters in clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, labels)
    silhouette_scores.append(silhouette_avg)

optimal_clusters = clusters_range[np.argmax(silhouette_scores)]
print(f"Optimal Number of Clusters: {optimal_clusters}")

# Visualize the clustering with K-Means
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
labels = kmeans.fit_predict(scaled_data)

# Plot the clustered data
plt.scatter(x, y, c=labels, cmap='viridis', marker=".", s=20)
plt.title(f'K-Means Clustering (k={optimal_clusters})')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Use TensorFlow for deep clustering (Autoencoder-based)
input_dim = data.shape[1]
encoding_dim = 2

# Build the autoencoder model
autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(encoding_dim, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(scaled_data, scaled_data, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Encode the data using the trained autoencoder
encoded_data = autoencoder.predict(scaled_data)

# Visualize the encoded data
plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=labels, cmap='viridis', marker=".", s=20)
plt.title('Deep Clustering with Autoencoder')
plt.xlabel('Encoded Dimension 1')
plt.ylabel('Encoded Dimension 2')
plt.show()
