import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image, ImageOps


def carregar_imagem(caminho):

    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(
            f"Imagem não encontrada: {caminho}"
        )

    imagem = Image.open(caminho)

    return imagem


def converter_para_cinza(imagem):

    return imagem.convert("L")


def inverter_cores(imagem):

    return ImageOps.invert(imagem)


def encontrar_bounding_box(imagem):

    array = np.array(imagem)

    pixels_digito = np.argwhere(array > 0)

    if pixels_digito.size == 0:
        raise ValueError(
            "Nenhum dígito foi identificado na imagem."
        )

    linha_min, coluna_min = pixels_digito.min(axis=0)
    linha_max, coluna_max = pixels_digito.max(axis=0)

    return (
        coluna_min,
        linha_min,
        coluna_max + 1,
        linha_max + 1
    )


def centralizar_digito(imagem, tamanho=28):

    bounding_box = encontrar_bounding_box(imagem)

    digito = imagem.crop(bounding_box)

    largura, altura = digito.size

    escala = min(
        (tamanho - 4) / largura,
        (tamanho - 4) / altura
    )

    nova_largura = max(1, int(largura * escala))
    nova_altura = max(1, int(altura * escala))

    digito = digito.resize(
        (nova_largura, nova_altura),
        Image.Resampling.LANCZOS
    )

    imagem_centralizada = Image.new(
        "L",
        (tamanho, tamanho),
        color=0
    )

    posicao_x = (tamanho - nova_largura) // 2
    posicao_y = (tamanho - nova_altura) // 2

    imagem_centralizada.paste(
        digito,
        (posicao_x, posicao_y)
    )

    return imagem_centralizada


def redimensionar_imagem(imagem, tamanho=(28, 28)):
   
    return imagem.resize(
        tamanho,
        Image.Resampling.LANCZOS
    )


def normalizar_imagem(imagem):

    array = np.array(imagem, dtype=np.float32)

    array = array / 255.0

    return array


def preprocessar_imagem(caminho):
  
    imagem = carregar_imagem(caminho)

    imagem = converter_para_cinza(imagem)

    imagem = inverter_cores(imagem)

    imagem = centralizar_digito(imagem)

    imagem = redimensionar_imagem(imagem)

    imagem_normalizada = normalizar_imagem(imagem)

    imagem_modelo = imagem_normalizada.reshape(1, -1)

    return imagem, imagem_normalizada, imagem_modelo


def prever_imagem(modelo, imagem_modelo):

    previsao = modelo.predict(imagem_modelo)

    probabilidades = modelo.predict_proba(imagem_modelo)

    digito_predito = previsao[0]

    probabilidades = probabilidades[0]

    return digito_predito, probabilidades


def plotar_previsao(
    imagem_processada,
    digito_predito,
    probabilidades,
    classes
):

    fig, eixos = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    eixos[0].imshow(
        imagem_processada,
        cmap="gray"
    )

    eixos[0].set_title(
        f"Imagem processada - Predição: {digito_predito}"
    )

    eixos[0].axis("off")

    eixos[1].bar(
        classes,
        probabilidades
    )

    eixos[1].set_title("Probabilidades por classe")
    eixos[1].set_xlabel("Dígito")
    eixos[1].set_ylabel("Probabilidade")

    eixos[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    plt.close(fig)