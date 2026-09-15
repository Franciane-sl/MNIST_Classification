from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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

def visualizar_digitos(x, y):
  
  fig, axes = plt.subplots(2, 5, figsize=(10, 5))

  fig.patch.set_facecolor("#DDA0DD")

  for digito in range(10):
     indice = np.where(y == str(digito))[0][0]
     imagem = x[indice].reshape(28, 28)

     axes.flat[digito].imshow(imagem, cmap="gray")
     axes.flat[digito].set_title(str(digito))
     axes.flat[digito].axis("off")

  plt.tight_layout()
  pasta_imagens = Path(__file__).resolve().parent.parent / "imagens" / "eda"
  pasta_imagens.mkdir(parents=True, exist_ok=True)

  caminho_imagem = pasta_imagens / "visualizacao_digitos.png"
  fig.savefig(caminho_imagem)

  plt.show()