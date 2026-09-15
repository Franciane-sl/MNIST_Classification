# MNIST Classification

Pipeline de Machine Learning para classificação de dígitos manuscritos utilizando o dataset **MNIST**, com comparação entre diferentes modelos de classificação e experimentos voltados à avaliação de **robustez e generalização**.

O projeto foi desenvolvido como um mini-projeto avaliativo do módulo de **Desenvolvimento de IA para Análise Preditiva**.

---

## Sobre o projeto

O projeto tem como objetivo desenvolver um pipeline completo de Machine Learning para reconhecimento de dígitos manuscritos.

A aplicação parte do carregamento e análise exploratória do dataset MNIST, realiza o pré-processamento dos dados, treina e compara três modelos de classificação e, posteriormente, investiga o comportamento dos modelos em situações diferentes das condições utilizadas durante o treinamento.

Além da avaliação tradicional no conjunto de teste, foram realizados experimentos para analisar:

* treinamento sem determinadas classes;
* comportamento diante de classes que não foram apresentadas durante o treinamento;
* confiança das previsões em situações fora do padrão;
* generalização para imagens manuscritas próprias.

O projeto foi estruturado de forma modular, separando as funções reutilizáveis do código de execução e análise presentes no notebook.

---

## Objetivos

### Objetivo geral

Desenvolver e avaliar um pipeline de classificação de imagens utilizando o dataset MNIST, comparando diferentes algoritmos de Machine Learning e investigando sua capacidade de generalização.

### Objetivos específicos

* Carregar e analisar o dataset MNIST;
* investigar a distribuição das classes;
* compreender a representação das imagens em formato vetorial;
* dividir os dados em treino, validação e teste utilizando estratificação;
* normalizar os valores dos pixels;
* implementar KNN, Random Forest e MLP;
* avaliar diferentes configurações de hiperparâmetros;
* comparar o desempenho dos modelos;
* analisar matrizes de confusão;
* medir o custo computacional de treinamento e predição;
* realizar Class Masking;
* testar generalização sobre classes ocultadas;
* analisar possíveis situações de overconfidence;
* realizar inferência sobre imagens manuscritas próprias.

---

## Dataset

Foi utilizado o dataset **MNIST**, carregado por meio do `fetch_openml` da biblioteca Scikit-Learn.

O dataset possui:

* **70.000 imagens**;
* **10 classes**, correspondentes aos dígitos de 0 a 9;
* imagens originais com dimensão **28 × 28 pixels**;
* **784 features** por imagem após a transformação em vetor;
* valores de intensidade dos pixels originalmente na escala **0 a 255**.

A representação utilizada pelos modelos é vetorial:

```text
28 × 28 = 784 pixels
```

Assim, cada imagem é transformada em um vetor contendo 784 valores.

### Distribuição das classes

A distribuição observada no dataset foi:

| Dígito | Quantidade |
| -----: | ---------: |
|      0 |      6.903 |
|      1 |      7.877 |
|      2 |      6.990 |
|      3 |      7.141 |
|      4 |      6.824 |
|      5 |      6.313 |
|      6 |      6.876 |
|      7 |      7.293 |
|      8 |      6.825 |
|      9 |      6.958 |

A classe 1 apresentou a maior quantidade de imagens, enquanto a classe 5 apresentou a menor. A diferença foi de 1.564 imagens, indicando uma distribuição relativamente equilibrada entre as classes.

---

# Pipeline do projeto

O desenvolvimento foi dividido em cinco fases:

```text
MNIST
  │
  ▼
Fase 1 — Carregamento e EDA
  │
  ▼
Fase 2 — Pré-processamento
  │
  ▼
Fase 3 — Treinamento dos modelos
  │
  ▼
Fase 4 — Avaliação comparativa
  │
  ▼
Fase 5 — Robustez e generalização
```

---

# Fase 1 — Carregamento e Análise Exploratória

A primeira etapa consistiu no carregamento e inspeção inicial do dataset.

Foram analisados:

* dimensões de `X` e `y`;
* quantidade de imagens;
* quantidade de features;
* quantidade de classes;
* distribuição dos dígitos;
* balanceamento das classes.

Também foi criada uma grade visual **2 × 5**, contendo um exemplo de cada dígito do MNIST.

A análise permitiu verificar que cada imagem possui dimensão original de 28 × 28 pixels e que os modelos recebem essas imagens como vetores de 784 features.

### Intensidade dos pixels

Os pixels do MNIST originalmente possuem valores entre **0 e 255**:

* `0` representa o preto;
* `255` representa o branco;
* valores intermediários representam diferentes níveis de intensidade de cinza.

A visualização dos exemplos também permitiu observar diferenças no estilo de escrita dos dígitos.

---

# Fase 2 — Pré-processamento e divisão dos dados

Os dados foram divididos utilizando estratificação por classe, garantindo que a distribuição dos dígitos fosse preservada nos diferentes conjuntos.

A divisão final foi:

| Conjunto  | Quantidade | Percentual |
| --------- | ---------: | ---------: |
| Treino    |     49.000 |        70% |
| Validação |      7.000 |        10% |
| Teste     |     14.000 |        20% |
| **Total** | **70.000** |   **100%** |

Foi utilizado `stratify` durante a divisão dos dados.

### Normalização

Os pixels foram normalizados de:

```text
[0, 255]
```

para:

```text
[0, 1]
```

utilizando:

```python
X / 255.0
```

A normalização foi realizada após a divisão dos dados, mantendo o conjunto de teste independente.

Essa transformação é importante porque coloca os atributos em uma escala uniforme, beneficiando principalmente algoritmos baseados em distância, como o KNN, além de facilitar o treinamento de modelos de redes neurais.

---

# Fase 3 — Implementação e treinamento dos modelos

Foram escolhidos três modelos de classificação com características diferentes:

1. **K-Nearest Neighbors (KNN)**
2. **Random Forest**
3. **Multi-Layer Perceptron (MLP)**

Para cada modelo foram avaliadas seis configurações de hiperparâmetros.

## KNN

Foram avaliados:

* `n_neighbors`: 3, 5 e 7;
* `weights`: `uniform` e `distance`.

A melhor configuração encontrada na validação foi:

```text
n_neighbors = 3
weights = distance
```

Resultado na validação:

```text
Acurácia: 97,23%
```

---

## Random Forest

Foram avaliados:

* `n_estimators`: 100, 200 e 300;
* `max_depth`: 10 e 20.

A melhor configuração encontrada foi:

```text
n_estimators = 300
max_depth = 20
```

Resultado na validação:

```text
Acurácia: 96,71%
```

---

## MLP

Foram avaliados:

* número de neurônios na camada oculta: 10, 20 e 30;
* `learning_rate_init`: 0.001 e 0.01.

A melhor configuração encontrada foi:

```text
hidden_layer_sizes = (30,)
learning_rate_init = 0.001
```

Resultado na validação:

```text
Acurácia: 96,13%
```

O conjunto de teste permaneceu reservado para a avaliação comparativa da Fase 4.

---

# Fase 4 — Avaliação comparativa

Os três modelos foram avaliados no conjunto de teste independente.

Foram utilizadas as seguintes métricas:

* Accuracy;
* Precision ponderada;
* Recall ponderado;
* F1-Score ponderado.

Também foram geradas matrizes de confusão 10 × 10 para cada modelo e medidos os tempos de treinamento e predição.

## Resultados

| Modelo        |   Accuracy |  Precision |     Recall |   F1-Score | Treino (s) | Predição (s) |
| ------------- | ---------: | ---------: | ---------: | ---------: | ---------: | -----------: |
| **KNN**       | **97,21%** | **97,24%** | **97,21%** | **97,21%** |   **0,48** |       169,30 |
| Random Forest |     96,61% |     96,61% |     96,61% |     96,61% |     285,57 |        47,97 |
| MLP           |     96,21% |     96,21% |     96,21% |     96,21% |     895,52 |     **3,01** |

### Análise das matrizes de confusão

A principal confusão encontrada nos três modelos foi entre os dígitos **4 e 9**, especificamente na situação em que um `4` real foi classificado como `9`.

| Modelo        | Maior confusão         |
| ------------- | ---------------------- |
| KNN           | 4 → 9 — 35 ocorrências |
| Random Forest | 4 → 9 — 38 ocorrências |
| MLP           | 4 → 9 — 35 ocorrências |

### Análise do custo computacional

O KNN apresentou a melhor performance preditiva e também o menor tempo de treinamento entre os três modelos. Entretanto, apresentou um custo muito maior durante a predição.

O Random Forest apresentou desempenho intermediário e um tempo de predição significativamente menor que o KNN.

O MLP apresentou o maior tempo de treinamento, porém obteve o menor tempo de predição.

Dessa forma, a escolha do modelo depende não apenas da acurácia, mas também do contexto computacional da aplicação.

Para os experimentos seguintes, o **KNN foi considerado o melhor modelo em desempenho preditivo**, apresentando 97,21% de acurácia no conjunto de teste.

---

# Fase 5 — Robustez e Generalização

A quinta fase teve como objetivo investigar o comportamento dos modelos em situações diferentes das condições ideais de treinamento.

Foram realizados três experimentos:

```text
Fase 5
│
├── 5.1 Class Masking
│
├── 5.2 Generalização extrema / OOD
│
└── 5.3 Imagens manuscritas próprias
```

---

## Fase 5.1 — Class Masking

As classes **4 e 7** foram completamente removidas do conjunto de treinamento.

O conjunto original de treinamento possuía:

```text
49.000 imagens
```

Após a remoção das classes:

```text
39.118 imagens
```

O modelo utilizado foi o **Random Forest**, configurado com:

```text
n_estimators = 300
max_depth = 20
```

As classes disponíveis durante o treinamento foram:

```text
0, 1, 2, 3, 5, 6, 8, 9
```

As classes `4` e `7` não participaram do ajuste do modelo.

O modelo treinado foi salvo em:

```text
modelos/random_forest_class_masking.pkl
```

Esse experimento permitiu criar uma situação controlada em que o classificador precisava lidar posteriormente com classes que não faziam parte do seu conjunto de treinamento.

---

## Fase 5.2 — Generalização extrema / OOD

Após o Class Masking, o Random Forest foi submetido exclusivamente às imagens das classes ocultadas:

* classe 4: 1.365 imagens;
* classe 7: 1.459 imagens;
* total: 2.824 imagens.

Como as classes 4 e 7 não faziam parte do conjunto de classes aprendidas, o modelo não possuía uma categoria específica para classificá-las como "desconhecidas".

Consequentemente, as imagens foram obrigatoriamente associadas a uma das classes conhecidas.

### Distribuição das previsões

Para as imagens reais da classe 4:

```text
93,11% foram classificadas como 9
```

Para as imagens reais da classe 7:

```text
74,09% foram classificadas como 9
```

Isso demonstra uma forte tendência do modelo de associar as classes desconhecidas à classe 9.

### Overconfidence

A análise das probabilidades mostrou que o modelo poderia apresentar elevada confiança mesmo estando diante de uma classe que não havia sido utilizada no treinamento.

A maior confiança observada foi:

```text
98,16%
```

Além disso:

* classe 4: 3,30% das previsões apresentaram confiança ≥ 90%;
* classe 7: 3,43% das previsões apresentaram confiança ≥ 90%.

O experimento demonstra uma limitação importante de classificadores tradicionais: **uma probabilidade elevada não significa necessariamente que a entrada pertence ao conjunto de classes conhecido pelo modelo**.

---

## Fase 5.3 — Inferência com imagens manuscritas próprias

Foram utilizadas dez imagens manuscritas próprias, correspondentes aos dígitos de 0 a 9.

As imagens foram armazenadas em:

```text
imagens/minhas_imagens/
```

### Pipeline de pré-processamento

Como as imagens foram fotografadas em papel branco com caneta escura, foi aplicado um pipeline específico:

```text
Imagem original
      ↓
Escala de cinza
      ↓
Aumento de contraste
      ↓
Inversão de cores
      ↓
Identificação do bounding box
      ↓
Centralização do dígito
      ↓
Redimensionamento para 28 × 28
      ↓
Normalização [0, 1]
      ↓
Vetor com 784 features
      ↓
Predição
```

O aumento de contraste foi utilizado para melhorar a definição dos dígitos após a conversão e inversão das imagens.

### Resultado

O melhor modelo da Fase 4, o KNN, foi utilizado para realizar as previsões.

Resultado obtido:

| Métrica           |  Resultado |
| ----------------- | ---------: |
| Imagens avaliadas |         10 |
| Acertos           |          0 |
| Erros             |         10 |
| Acurácia          |  **0,00%** |
| Confiança média   | **63,45%** |

Apesar da acurácia de 0%, algumas previsões apresentaram confiança elevada.

Dois exemplos importantes foram:

* dígito `6` → previsão `1` com **100% de confiança**;
* dígito `9` → previsão `1` com **100% de confiança**.

Esse resultado evidencia que o excelente desempenho obtido no conjunto MNIST não garante boa generalização para imagens produzidas fora da distribuição original do dataset.

---

# Principais resultados e conclusões

Os experimentos demonstraram que o projeto apresenta resultados elevados quando os dados de avaliação possuem características semelhantes às utilizadas no treinamento.

O **KNN apresentou a maior acurácia no conjunto de teste, com 97,21%**, sendo escolhido como melhor modelo em desempenho preditivo.

Entretanto, a análise do custo computacional mostrou que o KNN possui um tempo de predição muito superior ao Random Forest e ao MLP.

Os experimentos da Fase 5 mostraram uma limitação ainda mais importante: **bom desempenho no conjunto de teste não implica necessariamente boa capacidade de generalização**.

No Class Masking, o Random Forest conseguiu ser treinado sem as classes 4 e 7. Quando posteriormente recebeu somente imagens dessas classes, foi obrigado a classificá-las como alguma das categorias conhecidas.

Além disso, algumas dessas previsões apresentaram confiança elevada, caracterizando um comportamento de possível **overconfidence**.

Nas imagens manuscritas próprias, o KNN apresentou 0% de acurácia, mesmo tendo alcançado 97,21% no MNIST.

Esses resultados mostram a importância de avaliar modelos não apenas em condições semelhantes às do treinamento, mas também em cenários de **mudança de distribuição, classes desconhecidas e dados reais**.

---

# Tecnologias utilizadas

* **Python**
* **NumPy**
* **Pandas**
* **Scikit-Learn**
* **Matplotlib**
* **Seaborn**
* **Pillow**
* **Joblib**
* **Jupyter Notebook**

As versões utilizadas estão especificadas no arquivo `requirements.txt`.

---

# Estrutura do projeto

```text
MNIST_Classification/
│
├── imagens/
│   ├── eda/
│   ├── matrizes_de_confusao/
│   ├── minhas_imagens/
│   └── ood/
│
├── modelos/
│
├── notebooks/
│   └── mnist_classification.ipynb
│
├── src/
│   ├── __init__.py
│   ├── analise_exploratoria.py
│   ├── avaliacao.py
│   ├── carregamento.py
│   ├── imagens_proprias.py
│   ├── modelos.py
│   ├── robustez.py
│   └── tratamento_de_dados.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Organização dos módulos

| Arquivo                   | Responsabilidade                                  |
| ------------------------- | ------------------------------------------------- |
| `carregamento.py`         | Carregamento do dataset MNIST                     |
| `analise_exploratoria.py` | Inspeção dos dados e visualização dos dígitos     |
| `tratamento_de_dados.py`  | Divisão, verificação e normalização dos dados     |
| `modelos.py`              | Criação, treinamento e predição dos modelos       |
| `avaliacao.py`            | Métricas e matrizes de confusão                   |
| `robustez.py`             | Class Masking e análise OOD                       |
| `imagens_proprias.py`     | Pré-processamento e predição das imagens próprias |

O notebook `mnist_classification.ipynb` concentra a execução do pipeline, os experimentos, resultados e interpretações de cada fase.

---

# Como executar o projeto

## 1. Clonar o repositório

```bash
git clone https://github.com/Franciane-sl/MNIST_Classification.git
cd MNIST_Classification
```

## 2. Criar o ambiente virtual

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

## 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

## 4. Executar o notebook

Abra o projeto no VS Code e execute:

```text
notebooks/mnist_classification.ipynb
```

Selecione o ambiente virtual criado como kernel do notebook.

O dataset MNIST será carregado durante a execução por meio do `fetch_openml`.

---

# Reprodutibilidade

O projeto utiliza:

* `random_state=42` nas divisões dos dados;
* `random_state=42` nos modelos Random Forest e MLP;
* divisão estratificada dos conjuntos;
* `requirements.txt` com versões das dependências;
* caminhos relativos para os arquivos do projeto.

Essas práticas permitem reproduzir os experimentos em um ambiente configurado de forma equivalente.

---

# Organização das branches

O desenvolvimento foi organizado utilizando uma branch principal de desenvolvimento e branches específicas para cada etapa:

```text
main
 │
 └── develop
      │
      ├── feature/Estrutura_do_projeto
      │
      ├── feature/Carregamento_e_EDA
      │
      ├── feature/pre_processamento
      │
      ├── feature/modelos-preditivos
      │
      ├── feature/avaliacao_comparativa
      │
      ├── feature/robustez_generalizacao
      │
      └── feature/documentacao
```

Na branch feature/robustez-generalizacao concentra os experimentos da Fase 5:

* Class Masking;
* generalização extrema/OOD;
* imagens manuscritas próprias.

---

# Limitações do projeto

Os experimentos também evidenciaram algumas limitações:

* o KNN possui alto custo de predição;
* o MLP possui alto custo de treinamento;
* os modelos tradicionais utilizados não possuem um mecanismo explícito para rejeitar classes desconhecidas;
* as imagens manuscritas próprias apresentam uma distribuição diferente daquela utilizada no treinamento;
* o experimento OOD foi realizado por meio de classes ocultadas dentro do próprio MNIST, representando um cenário controlado de generalização.

---

# Possíveis melhorias

Como continuidade do projeto, poderiam ser exploradas:

* utilização de redes convolucionais (CNN);
* aumento e diversificação dos dados de treinamento;
* técnicas específicas de Open Set Recognition;
* mecanismos de rejeição para entradas desconhecidas;
* calibração das probabilidades dos modelos;
* utilização de datasets externos para avaliação OOD;
* aprimoramento do pré-processamento das imagens manuscritas;
* comparação com modelos adicionais, como SVM;
* análise mais aprofundada do custo computacional.

---

# Conclusão

O projeto demonstrou a construção de um pipeline completo de Machine Learning para classificação de imagens, desde o carregamento e análise exploratória até a avaliação de robustez e generalização.

A comparação entre KNN, Random Forest e MLP mostrou que o melhor desempenho preditivo foi obtido pelo KNN, com **97,21% de acurácia** no conjunto de teste.

Por outro lado, os experimentos de robustez mostraram que métricas elevadas em um conjunto de teste controlado não são suficientes para garantir generalização para dados desconhecidos ou com distribuição diferente.

A utilização de classes ocultadas e de imagens manuscritas próprias permitiu evidenciar situações de classificação forçada e de **alta confiança em previsões incorretas**, reforçando a importância de avaliar modelos de Inteligência Artificial também em cenários fora das condições ideais de treinamento.
