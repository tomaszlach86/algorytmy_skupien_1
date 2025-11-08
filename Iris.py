# --- K-Means clustering on Iris dataset ---

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# 1. Wczytanie danych
iris = load_iris()
X = iris.data  # tylko cechy (bez etykiet)
df = pd.DataFrame(X, columns=iris.feature_names)

# 2. Inicjalizacja modelu K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 3. Wyniki
df['cluster'] = kmeans.labels_

# 4. Redukcja wymiaru do 2D (dla wizualizacji)
pca = PCA(n_components=2)
reduced = pca.fit_transform(X)
df['x'] = reduced[:, 0]
df['y'] = reduced[:, 1]

# 5. Wizualizacja
plt.figure(figsize=(8, 6))
plt.scatter(df['x'], df['y'], c=df['cluster'], cmap='viridis')
plt.title('Klasteryzacja K-Means na zbiorze Iris')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# 6. Środki klastrów
print("Współrzędne centrów klastrów:\n", kmeans.cluster_centers_)
