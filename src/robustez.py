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
    classes_reais,
    classes_preditas
):

    matriz = np.zeros(
        (len(classes_reais), len(classes_preditas)),
        dtype=int
    )

    for i, classe_real in enumerate(classes_reais):

        for j, classe_predita in enumerate(classes_preditas):

            matriz[i, j] = np.sum(
                (y_real == classe_real) &
                (previsoes == classe_predita)
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

    return 

def analisar_confianca_ood(
    probabilidades,
    y_real
):
    confiancas = probabilidades.max(axis=1)

    resultados = {}

    classes_reais = np.unique(y_real)

    for classe in classes_reais:

        mascara = y_real == classe

        confiancas_classe = confiancas[mascara]

        resultados[classe] = {
            "confianca_media": float(
                confiancas_classe.mean()
            ),
            "confianca_minima": float(
                confiancas_classe.min()
            ),
            "confianca_maxima": float(
                confiancas_classe.max()
            )
        }

    return resultados

def contar_confiancas_ood(
    probabilidades,
    y_real,
    limiares=(0.80, 0.90, 0.95)
):
 
    confiancas = probabilidades.max(axis=1)

    resultados = {}

    for classe in np.unique(y_real):

        mascara = y_real == classe
        confiancas_classe = confiancas[mascara]

        total = len(confiancas_classe)

        resultados[classe] = {}

        for limiar in limiares:

            quantidade = np.sum(
                confiancas_classe >= limiar
            )

            resultados[classe][f">={int(limiar * 100)}%"] = {
                "quantidade": int(quantidade),
                "percentual": float(
                    quantidade / total * 100
                )
            }

    return resultados

def plotar_distribuicao_previsoes_ood(
    resultados,
    nome_arquivo="distribuicao_previsoes_ood.png"
):

    classes_reais = list(resultados.keys())

    classes_preditas = sorted({
        classe_predita
        for resultado in resultados.values()
        for classe_predita in resultado.keys()
    })

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(classes_preditas))
    largura = 0.35

    for i, classe_real in enumerate(classes_reais):

        percentuais = [
            resultados[classe_real]
            .get(classe_predita, {})
            .get("percentual", 0)
            for classe_predita in classes_preditas
        ]

        ax.bar(
            x + i * largura,
            percentuais,
            largura,
            label=f"Classe real {classe_real}"
        )

    ax.set_title("Distribuição das Previsões OOD")
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Percentual (%)")

    ax.set_xticks(
        x + largura * (len(classes_reais) - 1) / 2
    )

    ax.set_xticklabels(classes_preditas)

    ax.legend()

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