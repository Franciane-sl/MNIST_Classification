from sklearn.neighbors import KNeighborsClassifier
import time

def criar_knn(n_neighbors=5, weights="distance"):
    modelo = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights
    )

    return modelo

def treinar_modelo(modelo, X_treino, y_treino):
    inicio = time.time()

    modelo.fit(X_treino, y_treino)

    tempo_treinamento = time.time() - inicio

    return modelo, tempo_treinamento

def prever_modelo(modelo, X):
    inicio = time.time()

    previsoes = modelo.predict(X)

    tempo_predicao = time.time() - inicio

    return previsoes, tempo_predicao