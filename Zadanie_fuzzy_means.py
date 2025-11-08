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


def initialize_membership(n_samples, k, random_state=42):
    np.random.seed(random_state)
    U = np.random.rand(n_samples, k)
    # normalizacja wierszy tak, żeby sumowały się do 1
    U = U / U.sum(axis=1, keepdims=True)
    return U

def fuzzy_c_means(X, k, m=2.0, max_iters=20, epsilon=1e-4, random_state=42):
    n_samples, n_features = X.shape
    U = initialize_membership(n_samples, k, random_state=random_state)

    history_centroids = []
    history_U = []

    for it in range(max_iters):
        # 1. Centroidy
        Um = U ** m  # (n_samples, k)
        centroids = (Um.T @ X) / Um.sum(axis=0)[:, None]  # (k, n_features)

        history_centroids.append(centroids.copy())
        history_U.append(U.copy())

        # 2. Aktualizacja U
        distances = np.zeros((n_samples, k))
        for j in range(k):
            distances[:, j] = np.linalg.norm(X - centroids[j], axis=1)

        # obsługa dystansu 0 (punkt dokładnie w centroidzie)
        zero_mask = distances == 0
        if np.any(zero_mask):
            # jeżeli punkt pokrywa się z centroidem, to przynależność 1 do tego klastra, 0 do innych
            U_new = np.zeros_like(U)
            for i in range(n_samples):
                zeros_for_point = np.where(zero_mask[i])[0]
                if len(zeros_for_point) > 0:
                    j0 = zeros_for_point[0]
                    U_new[i, j0] = 1.0
                else:
                    # standardowa aktualizacja
                    for j in range(k):
                        denom = 0.0
                        for l in range(k):
                            ratio = distances[i, j] / distances[i, l]
                            denom += (ratio ** (2 / (m - 1)))
                        U_new[i, j] = 1.0 / denom
        else:
            U_new = np.zeros_like(U)
            for i in range(n_samples):
                for j in range(k):
                    denom = 0.0
                    for l in range(k):
                        ratio = distances[i, j] / distances[i, l]
                        denom += (ratio ** (2 / (m - 1)))
                    U_new[i, j] = 1.0 / denom

        # 3. Sprawdzenie zbieżności
        if np.max(np.abs(U_new - U)) < epsilon:
            U = U_new
            history_centroids.append(centroids.copy())
            history_U.append(U.copy())
            print(f"FCM: zbieżność po {it+1} iteracjach")
            break

        U = U_new

    return centroids, history_centroids, history_U

def plot_fcm_iteration(X, centroids, U, iteration, k):
    hard_labels = np.argmax(U, axis=1)
    max_membership = np.max(U, axis=1)

    plt.figure(figsize=(6, 5))
    for j in range(k):
        mask = hard_labels == j
        sizes = 50 + 250 * max_membership[mask]
        plt.scatter(X[mask, 0], X[mask, 1],
                    s=sizes, alpha=0.7, label=f'Klaster {j}')
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='X', s=200, edgecolor='black', linewidths=1.5,
                label='Centroidy', c='red')
    plt.title(f'Fuzzy C-Means: iteracja {iteration}, k = {k}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def fcm_tables(X, history_centroids, history_U):
    point_names = [f'P{i}' for i in range(1, len(X)+1)]
    for it, (centroids, U) in enumerate(zip(history_centroids, history_U), start=1):
        print(f'\n=== FCM – Iteracja {it} ===')

        # Tabela centroidów
        df_cent = pd.DataFrame(centroids, columns=['cx', 'cy'])
        df_cent.index = [f'C{j}' for j in range(len(centroids))]
        print('\nCentroidy:')
        print(df_cent)

        # Macierz przynależności U
        cols = [f'klaster_{j}' for j in range(U.shape[1])]
        df_U = pd.DataFrame(U, columns=cols)
        df_U.insert(0, 'punkt', point_names)
        print('\nMacierz przynależności U:')
        print(df_U)

        # (opcjonalnie) wykres
        plot_fcm_iteration(X, centroids, U, it, k=U.shape[1])


X = get_data(with_outlier=False)
# X = get_data(with_outlier=True)

for k in [2, 3]:
    print(f'\n################ FCM dla k = {k}, m = 2.0 ################')
    final_centroids, hist_centroids, hist_U = fuzzy_c_means(
        X, k=k, m=2.0, max_iters=20, epsilon=1e-4, random_state=42
    )
    fcm_tables(X, hist_centroids, hist_U)
