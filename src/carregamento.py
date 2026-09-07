from sklearn.datasets import fetch_openml


"Função para carregar o dataset."

def carregar_dataset():
    
    print("Baixando o dataset MNIST...")

    try:
        mnist = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            parser="auto"
        )

        X = mnist.data
        y = mnist.target

        print("O Dataset MNIST foi carregado com sucesso!")

        return X, y

    except Exception as erro:
        raise RuntimeError(
            f"Não foi possível carregar o dataset MNIST: {erro}"
        ) from erro