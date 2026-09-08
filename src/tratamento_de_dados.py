from sklearn.model_selection import train_test_split


def dividir_dados(X, y):

    X_treino_validacao, X_teste, y_treino_validacao, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    X_treino, X_validacao, y_treino, y_validacao = train_test_split(
        X_treino_validacao,
        y_treino_validacao,
        test_size=0.125,
        stratify=y_treino_validacao,
        random_state=42
    )

    return (
        X_treino,
        X_validacao,
        X_teste,
        y_treino,
        y_validacao,
        y_teste
    )

def verificar_dimensoes(
    X_treino,
    X_validacao,
    X_teste,
    y_treino,
    y_validacao,
    y_teste
):

    print("Dimensões dos conjuntos:")

    print(f"X treino: {X_treino.shape}")
    print(f"X validação: {X_validacao.shape}")
    print(f"X teste: {X_teste.shape}")

    print("\nDimensões dos rótulos:")

    print(f"y treino: {y_treino.shape}")
    print(f"y validação: {y_validacao.shape}")
    print(f"y teste: {y_teste.shape}")

def verificar_distribuicao_classes(y_treino, y_validacao, y_teste):

    classes = sorted(set(y_treino))

    print("Distribuição das classes:\n")

    print(
        f"{'Classe':<10}"
        f"{'Treino':<12}"
        f"{'% Treino':<12}"
        f"{'Validação':<14}"
        f"{'% Validação':<14}"
        f"{'Teste':<12}"
        f"{'% Teste':<10}"
    )

    for classe in classes:
        quantidade_treino = sum(y == classe for y in y_treino)
        quantidade_validacao = sum(y == classe for y in y_validacao)
        quantidade_teste = sum(y == classe for y in y_teste)

        percentual_treino = quantidade_treino / len(y_treino) * 100
        percentual_validacao = quantidade_validacao / len(y_validacao) * 100
        percentual_teste = quantidade_teste / len(y_teste) * 100

        print(
            f"{classe:<10}"
            f"{quantidade_treino:<12}"
            f"{percentual_treino:<12.2f}"
            f"{quantidade_validacao:<14}"
            f"{percentual_validacao:<14.2f}"
            f"{quantidade_teste:<12}"
            f"{percentual_teste:<10.2f}"
        )

def normalizar_dados(X_treino, X_validacao, X_teste):
    
    X_treino_normalizado = X_treino / 255.0
    X_validacao_normalizado = X_validacao / 255.0
    X_teste_normalizado = X_teste / 255.0

    return (
        X_treino_normalizado,
        X_validacao_normalizado,
        X_teste_normalizado
    )

def verificar_normalizacao(X_treino, X_validacao, X_teste):

    print("Verificação da normalização:\n")

    print("X treino:")
    print(f"Valor mínimo: {X_treino.min():.2f}")
    print(f"Valor máximo: {X_treino.max():.2f}")
    print(f"Dimensão: {X_treino.shape}")

    print("\nX validação:")
    print(f"Valor mínimo: {X_validacao.min():.2f}")
    print(f"Valor máximo: {X_validacao.max():.2f}")
    print(f"Dimensão: {X_validacao.shape}")

    print("\nX teste:")
    print(f"Valor mínimo: {X_teste.min():.2f}")
    print(f"Valor máximo: {X_teste.max():.2f}")
    print(f"Dimensão: {X_teste.shape}")