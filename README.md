# Algoritmos de Regressão e Classificação

Implementações de algoritmos de aprendizado de máquina para o primeiro trabalho de Machine Learning.

## 📋 Descrição

Este projeto contém implementações de quatro algoritmos fundamentais de aprendizado de máquina:

1. **Regressão Linear** - Para predição de valores contínuos
2. **Regressão Logística** - Para classificação binária e multiclasse
3. **K-Nearest Neighbors (KNN)** - Para classificação e regressão baseada em vizinhos
4. **Redes Neurais** - Para aprendizado profundo e reconhecimento de padrões complexos

## 🗂️ Estrutura do Projeto

```
.
├── src/
│   ├── algorithms/          # Implementações dos algoritmos
│   │   ├── linear_regression.py
│   │   ├── logistic_regression.py
│   │   ├── knn.py
│   │   └── neural_network.py
│   └── data/               # Carregamento de datasets
│       └── datasets.py
├── tests/                  # Testes unitários
│   ├── test_algorithms.py
│   └── test_datasets.py
├── main.py                 # Ponto de entrada principal
└── requirements.txt        # Dependências do projeto
```

## 🚀 Configuração do Ambiente

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/billygrahan/Algoritmos_de_Regre-o_e_Classifica-o.git
cd Algoritmos_de_Regre-o_e_Classifica-o
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 📊 Datasets Disponíveis

O projeto utiliza dois datasets do scikit-learn, carregados através do pandas:

### 1. California Housing Dataset
- **Tipo**: Regressão
- **Amostras**: 20,640
- **Features**: 8
- **Descrição**: Dados de preços de casas na Califórnia

### 2. Breast Cancer Dataset
- **Tipo**: Classificação Binária
- **Amostras**: 569
- **Features**: 30
- **Classes**: 2 (Maligno/Benigno)
- **Descrição**: Dados de diagnóstico de câncer de mama

## 🎯 Uso

### Executar o programa principal:
```bash
python main.py
```

### Executar os testes:
```bash
pytest tests/
```

### Exemplo de uso dos datasets:
```python
from src.data import load_california_housing, load_breast_cancer

# Carregar California Housing
X_calif, y_calif = load_california_housing(return_X_y=True)

# Carregar Breast Cancer
X_cancer, y_cancer = load_breast_cancer(return_X_y=True)
```

### Exemplo de uso dos algoritmos:
```python
from src.algorithms import LinearRegression, LogisticRegression, KNN, NeuralNetwork

# Criar instâncias dos modelos
lr = LinearRegression()
logr = LogisticRegression()
knn = KNN(k=5)
nn = NeuralNetwork(hidden_layers=(100,))

# Os métodos fit() e predict() serão implementados nas próximas iterações
```

## 🧪 Testes

O projeto inclui testes unitários para:
- Carregamento de datasets
- Inicialização dos algoritmos
- Validação de parâmetros

Execute os testes com:
```bash
pytest tests/ -v
```

## 📦 Dependências

- `numpy>=1.24.0` - Computação numérica
- `pandas>=2.0.0` - Manipulação de dados
- `scikit-learn>=1.3.0` - Datasets e utilitários ML
- `matplotlib>=3.7.0` - Visualização de dados
- `pytest>=7.4.0` - Framework de testes

## 🔧 Status do Projeto

- [x] Configuração do ambiente Python
- [x] Estrutura do projeto organizada
- [x] Carregamento de datasets (California Housing e Breast Cancer)
- [x] Estrutura base dos algoritmos
- [ ] Implementação da Regressão Linear
- [ ] Implementação da Regressão Logística
- [ ] Implementação do KNN
- [ ] Implementação das Redes Neurais
- [ ] Experimentos e avaliação dos modelos

## 👥 Autores

Billy Grahan

## 📄 Licença

Este projeto é parte de um trabalho acadêmico de Aprendizado de Máquina.
