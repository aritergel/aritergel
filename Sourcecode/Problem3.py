import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("results.csv")


numeric_df = df.select_dtypes(include=[np.number]).copy()
numeric_df = numeric_df.dropna(axis=1, thresh=len(numeric_df) * 0.7)  # drop columns with too many NaNs
numeric_df = numeric_df.fillna(numeric_df.mean())  # fill remaining NaNs


scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)


inertia = []
K_range = range(1, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_data)
    inertia.append(km.inertia_)


plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method to Determine k")
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_plot.png")
plt.close()
print("📈 Saved: elbow_plot.png (use to choose k)")


optimal_k = 4  
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)
df['Cluster'] = clusters


pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]


plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title(f"K-Means Clusters of Players (k={optimal_k})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig("clusters_pca_2D.png")
plt.close()
print("🎯 Saved: clusters_pca_2D.png")


print(f"\n🔎 Number of players in each cluster (k={optimal_k}):")
print(df['Cluster'].value_counts().sort_index())

