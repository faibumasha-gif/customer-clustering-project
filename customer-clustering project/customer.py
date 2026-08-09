# ==========================================
# CUSTOMER SEGMENTATION USING K-MEANS
# ==========================================
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("Mall_Customers.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ----------------------------
# Handle Missing Values
# ----------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)

# ----------------------------
# Encode Gender
# ----------------------------
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

# ----------------------------
# Select Features
# ----------------------------
X = df[["Age","Annual Income (k$)","Spending Score (1-100)"]]

# ----------------------------
# Feature Scaling
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# Exploratory Data Analysis
# ============================

# Age Distribution
plt.figure(figsize=(6,4))
plt.hist(df["Age"], bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Annual Income Distribution
plt.figure(figsize=(6,4))
plt.hist(df["Annual Income (k$)"], bins=10)
plt.title("Annual Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
plt.show()

# Spending Score Distribution
plt.figure(figsize=(6,4))
plt.hist(df["Spending Score (1-100)"], bins=10)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Count")
plt.show()

# ============================
# Elbow Method
# ============================

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# ============================
# KMeans Model
# ============================

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# ============================
# Silhouette Score
# ============================

score = silhouette_score(X_scaled, clusters)

print("\nSilhouette Score:", score)

# ============================
# Cluster Visualization
# ============================

plt.figure(figsize=(8,6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"]
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")

plt.show()

# ============================
# Cluster Centers
# ============================

centers = scaler.inverse_transform(kmeans.cluster_centers_)

center_df = pd.DataFrame(
    centers,
    columns=["Age","Annual Income","Spending Score"]
)

print("\nCluster Centers")
print(center_df)

# ============================
# Customer Count
# ============================

print("\nCustomers in Each Cluster")

print(df["Cluster"].value_counts().sort_index())

# ============================
# Cluster Summary
# ============================

summary = df.groupby("Cluster")[["Age",
"Annual Income (k$)",
"Spending Score (1-100)"]].mean()

print("\nCluster Summary")
print(summary)

print("\nProject Completed Successfully")