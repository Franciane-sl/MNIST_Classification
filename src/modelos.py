from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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

def criar_random_forest(n_estimators=100, max_depth=10):
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    return modelo

def criar_mlp(hidden_layer_sizes=(10,), learning_rate_init=0.001):
    modelo = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        random_state=42
    )
    return modelo