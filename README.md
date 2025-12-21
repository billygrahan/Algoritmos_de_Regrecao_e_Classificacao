# Algoritmos_de_Regrecao_e_Classificacao
Algoritmos de Regressão e classificação para o primeiro trabalho de aprendizado de máquina.

## Como executar o projeto

### 1. Preparar o ambiente

Dentro da pasta do projeto:

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

O arquivo de dependências é o [requirements.txt](requirements.txt).

---

### 2. Executar cada experimento

#### KNN (classificação)

Script principal: [KNN/Teste_knn.py](KNN/Teste_knn.py)  
Algoritmo KNN: [`KNN.Knn.knn`](KNN/Knn.py)

```bash
python KNN/Teste_knn.py
```

O script:
- Carrega a base de câncer de mama via [`Bases.Base_breast_cancer.data`](Bases/Base_breast_cancer.py)
- Separa treino/teste (80% / 20%)
- Calcula métricas (acurácia, precisão, recall, F1)
- Mostra matriz de confusão e gráficos de desempenho

---

#### Rede Neural (Backpropagation)

Script principal: [RN-Backprop/Teste_rnb.py](RN-Backprop/Teste_rnb.py)  
Modelo: [`RN-Backprop.Rnb.RedeNeural`](RN-Backprop/Rnb.py)

```bash
python RN-Backprop/Teste_rnb.py
```

O script:
- Carrega e normaliza os dados com [`RN-Backprop.Teste_rnb.carregar_dados`](RN-Backprop/Teste_rnb.py)
- Divide a base em 80% treino e 20% teste
- Treina a rede com backpropagation usando mini‑batch SGD
- Plota a evolução do erro de treino (MSE) e salva em `Plots/Backpropagation/erro_treino.png`
- Calcula matriz de confusão e métricas (acurácia, precisão, recall, F1)
- Gera e salva:
  - `Plots/Backpropagation/matriz_confusao.png`
  - `Plots/Backpropagation/metricas_desempenho.png`

---

#### Regressão Logística

Implementação em: [Regrecao_Logistica/RLog.py](Regrecao_Logistica/RLog.py)  
Funções principais: [`Regrecao_Logistica.RLog.treinar`](Regrecao_Logistica/RLog.py), [`Regrecao_Logistica.RLog.prever`](Regrecao_Logistica/RLog.py)

Exemplo simples de uso interativo:

```bash
python
```

```python
from Bases.Base_breast_cancer import data
from Regrecao_Logistica.RLog import treinar, prever

X = data.data.to_numpy()
y = data.target.to_numpy()

w, b = treinar(X, y, tx_aprendizado=0.01, ciclos=1000)
classe, prob = prever(X[0], w, b)
print(classe, prob)
```

---

#### Regressão Linear

Notebook: [Regrecao_Linear/rl.ipynb](Regrecao_Linear/rl.ipynb)

Abra o notebook no VS Code ou Jupyter e execute as células em ordem.  
Ele:
- Carrega o dataset California Housing (`sklearn.datasets.fetch_california_housing`)
- Normaliza os dados
- Treina uma regressão linear usando gradiente descendente
- Calcula MSE de teste e plota predições vs valores reais

---

## Como funciona o projeto

### Conjunto de dados

A base principal de classificação é carregada em [`Bases.Base_breast_cancer.data`](Bases/Base_breast_cancer.py) usando `sklearn.datasets.load_breast_cancer(as_frame=True)`.  
Para regressão linear é usado o dataset California Housing dentro do notebook.

### KNN

- Implementado em [`KNN.Knn.knn`](KNN/Knn.py).
- Calcula a distância euclidiana entre o ponto de teste e os pontos de treino.
- Escolhe os $k$ vizinhos mais próximos e faz votação de maioria para decidir a classe.

### Regressão Logística

- Implementada em [Regrecao_Logistica/RLog.py](Regrecao_Logistica/RLog.py).
- Usa função sigmoide $\sigma(z) = \frac{1}{1 + e^{-z}}$ para mapear $z = w^T x + b$ em probabilidade.
- Os parâmetros $w$ e $b$ são ajustados por gradiente descendente minimizando o custo logístico.

### Rede Neural (Backpropagation)

- Classe [`RN-Backprop.Rnb.RedeNeural`](RN-Backprop/Rnb.py) no estilo do livro de Michael Nielsen:
  - Camada de entrada (número de atributos)
  - Uma camada oculta com ativação sigmoide
  - Um neurônio de saída sigmoide
  - Pesos e biases armazenados em listas `self.weights` e `self.biases`.
- Treinamento com **mini‑batch SGD**:
  - O método `treinar` embaralha os dados, cria mini‑batches e chama `update_mini_batch`.
  - `backprop` calcula o gradiente do erro em relação a todos os pesos e biases.
  - `update_mini_batch` aplica a atualização de gradiente médio em cada mini‑batch.
- Funções principais:
  - `feedforward(a)`: propaga a entrada pela rede.
  - `backprop(x, y)`: retorna gradientes `(nabla_b, nabla_w)` para uma amostra.
  - `prever_proba(x)`: devolve a probabilidade da classe positiva.
  - `prever_classe(x, limiar)`: converte probabilidade em rótulo 0/1.
- Métrica de erro usada no treino: MSE
  \[
    \text{MSE} = \frac{1}{2}\|a - y\|^2
  \]
  e o script salva o erro médio por época em um gráfico.

### Regressão Linear

- Implementada no notebook [Regrecao_Linear/rl.ipynb](Regrecao_Linear/rl.ipynb).
- Modelo linear $y = w^T x + b$.
- Treino por gradiente descendente sobre o erro quadrático médio (MSE).
- Ao final, o notebook calcula o MSE em teste e plota $y_{\text{real}}$ vs $y_{\text{pred}}$.