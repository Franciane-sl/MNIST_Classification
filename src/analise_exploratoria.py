import numpy as np

"Função para inspecionar os dados do dataset."

def inspecionar_dados(x, y):

    print("\n" + "=" * 30 + " Inspeção inicial do dataset " + "=" * 30)
    print("\n" + "=" * 30 + " Dimensão de x" + "=" * 30)
    print(x.shape)
    print("\n" + "=" * 30 + " Dimensão de y" + "=" * 30)
    print(y.shape)
    print("\n" + "=" * 30 + " Quantidade de imagens " + "=" * 30)
    print(x.shape[0])
    print("\n" + "=" * 30 + " Quantidade de features por imagem " + "=" * 30)
    print(x.shape[1])
    print("\n" + "=" * 30 + " Quantidade de classes " + "=" * 30)
    print(len(np.unique(y)))
    print("\n"+ "=" * 30 + " Distribuição das classes " + "=" * 30)

    classes, quantidades = np.unique(y, return_counts=True)

    for classe, quantidade in zip(classes, quantidades):
        print(f"Classe {classe}: {quantidade} imagens")

    print("\n" + "=" * 30 + " Análise do balanceamento " + "=" * 30)

    indice_maior = np.argmax(quantidades)
    indice_menor = np.argmin(quantidades)

    print(f"Classe com mais imagens: {classes[indice_maior]} ({quantidades[indice_maior]} imagens)")
    print(f"Classe com menos imagens: {classes[indice_menor]} ({quantidades[indice_menor]} imagens)")

    diferenca = quantidades[indice_maior] - quantidades[indice_menor]

    print(f"Diferença entre as classes: {diferenca} imagens")
    print("\n" + "=" * 30 + " Fim da inspeção inicial do dataset " + "=" * 30)