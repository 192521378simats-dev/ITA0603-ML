import numpy as np
from sklearn.mixture import GaussianMixture

# Sample data
X = np.array([[1], [2], [3], [8], [9], [10]])

# Create EM model with 2 clusters
model = GaussianMixture(n_components=2, random_state=0)

# Train the model using EM algorithm
model.fit(X)

# Predict clusters
labels = model.predict(X)

print("Data:", X.flatten())
print("Cluster Labels:", labels)
print("Cluster Means:", model.means_.flatten())
print("Cluster Weights:", model.weights_)