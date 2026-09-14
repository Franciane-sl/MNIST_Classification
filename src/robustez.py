import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import confusion_matrix


def remover_classes_treinamento(
    X_treino,
    y_treino,
    classes_remover
):

    mascara = ~np.isin(y_treino, classes_remover)

    X_filtrado = X_treino[mascara]
    y_filtrado = y_treino[mascara]

    return X_filtrado, y_filtrado


def verificar_classes(y):

    return np.unique(y)


def selecionar_classes_teste(
    X_teste,
    y_teste,
    classes_selecionadas
):

    mascara = np.isin(y_teste, classes_selecionadas)

    X_filtrado = X_teste[mascara]
    y_filtrado = y_teste[mascara]

    return X_filtrado, y_filtrado


def obter_probabilidades(modelo, X):

    probabilidades = modelo.predict_proba(X)

    return probabilidades


def analisar_previsoes_ood(y_real, previsoes):
   
    resultados = {}

    classes_reais = np.unique(y_real)

    for classe in classes_reais:

        mascara = y_real == classe

        previsoes_classe = previsoes[mascara]

        classes, quantidades = np.unique(
            previsoes_classe,
            return_counts=True
        )

        total = len(previsoes_classe)

        distribuicao = {}

        for classe_predita, quantidade in zip(
            classes,
            quantidades
        ):
            distribuicao[classe_predita] = {
                "quantidade": int(quantidade),
                "percentual": float(
                    quantidade / total * 100
                )
            }

        resultados[classe] = distribuicao

    return resultados


def calcular_matriz_confusao_ood(
    y_real,
    previsoes,
    classes_reais
):

    matriz = confusion_matrix(
        y_real,
        previsoes,
        labels=classes_reais
    )

    return matriz


def plotar_matriz_confusao_ood(
    matriz,
    classes_reais,
    classes_preditas,
    nome_arquivo="matriz_confusao_ood.png"
):

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes_preditas,
        yticklabels=classes_reais,
        ax=ax
    )

    ax.set_title("Matriz de Confusão - OOD")
    ax.set_xlabel("Dígito Predito")
    ax.set_ylabel("Dígito Real")

    plt.tight_layout()

    pasta_imagens = (
        Path(__file__).resolve().parent.parent
        / "imagens"
        / "ood"
    )

    pasta_imagens.mkdir(
        parents=True,
        exist_ok=True
    )

    caminho_imagem = (
        pasta_imagens
        / nome_arquivo
    )

    fig.savefig(caminho_imagem)

    plt.show()

    plt.close(fig)

    return caminho_imagem