import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('../data/Mall_Customers.csv')

# Wybieramy tylko cechy numeryczne
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(8,6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
            c=df['cluster'], cmap='viridis')
plt.xlabel('Dochód roczny (k$)')
plt.ylabel('Wskaźnik wydatków')
plt.title('Segmentacja klientów centrum handlowego')
plt.show()
