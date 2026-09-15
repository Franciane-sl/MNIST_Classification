import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def calcular_metricas(y_teste, previsoes):

    metricas = {
        "Acurácia": accuracy_score(y_teste, previsoes),
        "Precisão": precision_score(
            y_teste,
            previsoes,
            average="weighted"
        ),
        "Recall": recall_score(
            y_teste,
            previsoes,
            average="weighted"
        ),
        "F1-score": f1_score(
            y_teste,
            previsoes,
            average="weighted"
        )
    }

    return metricas


def gerar_matriz_confusao(y_teste, previsoes):
   
    matriz = confusion_matrix(
        y_teste,
        previsoes,
        labels=np.array([str(i) for i in range(10)])
    )

    return matriz


def plotar_matriz_confusao(matriz, nome_modelo):

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.arange(10),
        yticklabels=np.arange(10),
        ax=ax
    )

    ax.set_title(f"Matriz de Confusão - {nome_modelo}")
    ax.set_xlabel("Dígito Predito")
    ax.set_ylabel("Dígito Real")

    plt.tight_layout()

    pasta_imagens = (
        Path(__file__).resolve().parent.parent
        / "imagens"
        / "matrizes_de_confusão"
    )

    pasta_imagens.mkdir(parents=True, exist_ok=True)

    nome_arquivo = (
        nome_modelo.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    caminho_imagem = (
        pasta_imagens
        / f"matriz_confusao_{nome_arquivo}.png"
    )

    fig.savefig(caminho_imagem)

    plt.show()

    plt.close(fig)


def identificar_maior_confusao(matriz):

    matriz_sem_diagonal = matriz.copy()

    np.fill_diagonal(matriz_sem_diagonal, 0)

    indice = np.unravel_index(
        np.argmax(matriz_sem_diagonal),
        matriz_sem_diagonal.shape
    )

    return {
        "digito_real": int(indice[0]),
        "digito_predito": int(indice[1]),
        "quantidade": int(matriz_sem_diagonal[indice])
    }