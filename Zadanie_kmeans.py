import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_data(with_outlier=False):
    points = np.array([
        [1, 2],     # P1
        [2, 1],     # P2
        [1.5, 1.8], # P3
        [5, 8],     # P4
        [6, 8],     # P5
        [5.5, 7.5], # P6
        [9, 1],     # P7
        [8, 2],     # P8
        [9, 2],     # P9
        [8.5, 1.5]  # P10 normalnie
    ], dtype=float)

    if with_outlier:
        points[-1] = np.array([18.5, 11.5])  # odchylenie P10

    return points




def kmeans(X, k, max_iters=10, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]

    # 1. Inicjalizacja centroidów – losujemy k różnych punktów
    initial_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[initial_indices].copy()

    history_centroids = []
    history_labels = []

    for it in range(max_iters):
        # 2. Przypisanie do najbliższego centroidu
        # odległości: (n_samples, k)
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        history_centroids.append(centroids.copy())
        history_labels.append(labels.copy())

        # 3. Aktualizacja centroidów
        new_centroids = centroids.copy()
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)
            # jeśli klaster jest pusty – zostaw stary centroid

        # 4. Sprawdzenie zbieżności
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            history_centroids.append(centroids.copy())
            history_labels.append(labels.copy())
            print(f"K-means: zbieżność po {it+1} iteracjach")
            break

        centroids = new_centroids

    return centroids, history_centroids, history_labels

def plot_kmeans_iteration(X, centroids, labels, iteration, k):
    plt.figure(figsize=(6, 5))
    for j in range(k):
        cluster_points = X[labels == j]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    label=f'Klaster {j}', alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='X', s=200, edgecolor='black', linewidths=1.5,
                label='Centroidy', c='red')
    plt.title(f'K-means: iteracja {iteration}, k = {k}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def kmeans_tables(X, history_centroids, history_labels):
    point_names = [f'P{i}' for i in range(1, len(X)+1)]
    for it, (centroids, labels) in enumerate(zip(history_centroids, history_labels), start=1):
        print(f'\n=== K-means – Iteracja {it} ===')
        # Tabela centroidów
        df_cent = pd.DataFrame(centroids, columns=['cx', 'cy'])
        df_cent.index = [f'C{j}' for j in range(len(centroids))]
        print('\nCentroidy:')
        print(df_cent)

        # Tabela przypisań punktów
        df_labels = pd.DataFrame({'punkt': point_names, 'klaster': labels})
        print('\nPrzypisania punktów:')
        print(df_labels)

        # (opcjonalnie) Wykres
        plot_kmeans_iteration(X, centroids, labels, it, k=len(centroids))


X = get_data(with_outlier=False)   # bez punktu odstającego
# X = get_data(with_outlier=True)  # z punktem odstającym

for k in [2, 3]:
    print(f'\n################ K-means dla k = {k} ################')
    final_centroids, hist_centroids, hist_labels = kmeans(X, k=k, max_iters=10)
    kmeans_tables(X, hist_centroids, hist_labels)
