import sys
sys.path.append('../Bases')
sys.path.append('../KNN')

from Base_breast_cancer import data
from Knn import knn
# import random

X = data.data.values.tolist()
y = data.target.values.tolist()

dados_combinados = list(zip(X, y))
# random.seed(42)  # Para reprodutibilidade
# random.shuffle(dados_combinados)

# Separa novamente X e y
X, y = zip(*dados_combinados)
X = list(X)
y = list(y)

# Divide em 80% treino e 20% teste
tamanho_total = len(X)
tamanho_treino = int(0.8 * tamanho_total)

X_treino = X[:tamanho_treino]
y_treino = y[:tamanho_treino]
X_teste = X[tamanho_treino:]
y_teste = y[tamanho_treino:]

print(f"Total de amostras: {tamanho_total}")
print(f"Amostras de treino: {len(X_treino)}")
print(f"Amostras de teste: {len(X_teste)}")
print()

# Testa com k=3
k = 3
print(f"Testando KNN com k={k}...")

acertos = 0
total_testes = len(X_teste)

for i, ponto_teste in enumerate(X_teste):
    predicao = knn(X_treino, y_treino, ponto_teste, k)
    if predicao == y_teste[i]:
        acertos += 1
    
    # if (i + 1) % 10 == 0:
    #     print(f"Testados: {i + 1}/{total_testes}")

acuracia = (acertos / total_testes) * 100

print()
print(f"Resultados:")
print(f"Total de testes: {total_testes}")
print(f"Acertos: {acertos}")
print(f"Erros: {total_testes - acertos}")
print(f"Acurácia: {acuracia:.2f}%")
