import sys
import numpy as np

sys.path.append('../Bases')
sys.path.append('../Regrecao_Logistica')

from Base_breast_cancer import data
from RL import treinar, prever

# IMPORTANTE: Converter para numpy array primeiro
X = data.data.values
y = data.target.values

# NORMALIZAR OS DADOS (essencial para convergência!)
media = X.mean(axis=0)
desvio_padrao = X.std(axis=0)
X = (X - media) / desvio_padrao

# Converter para lista depois da normalização
X = X.tolist()
y = y.tolist()

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

acertos = 0
total_testes = len(X_teste)

# Variáveis para a matriz de confusão
TP = 0  # True Positive: previu 1 e era 1
TN = 0  # True Negative: previu 0 e era 0
FP = 0  # False Positive: previu 1 mas era 0
FN = 0  # False Negative: previu 0 mas era 1

# Treinar com taxa de aprendizado maior (dados normalizados permitem isso)
w, b = treinar(np.array(X_treino), np.array(y_treino), 0.1, 1000)

for i, ponto_teste in enumerate(X_teste):
    predicao, probabilidade = prever(ponto_teste, w, b)
    real = y_teste[i]
    
    if predicao == real:
        acertos += 1
    
    # Preencher matriz de confusão
    if real == 1 and predicao == 1:
        TP += 1
    elif real == 0 and predicao == 0:
        TN += 1
    elif real == 0 and predicao == 1:
        FP += 1
    elif real == 1 and predicao == 0:
        FN += 1

acuracia = (acertos / total_testes) * 100

# Calcular métricas
precisao = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0

print()
print(f"========== RESULTADOS ==========")
print(f"Total de testes: {total_testes}")
print(f"Acertos: {acertos}")
print(f"Erros: {total_testes - acertos}")
print(f"Acurácia: {acuracia:.2f}%")
print()
print(f"========== MATRIZ DE CONFUSÃO ==========")
print(f"              Predito")
print(f"              0    1")
print(f"Real    0  [{TN:4d} {FP:4d}]")
print(f"        1  [{FN:4d} {TP:4d}]")
print()
print(f"TN (True Negative):  {TN} - Previu 0 e era 0 ✓")
print(f"FP (False Positive): {FP} - Previu 1 mas era 0 ✗")
print(f"FN (False Negative): {FN} - Previu 0 mas era 1 ✗")
print(f"TP (True Positive):  {TP} - Previu 1 e era 1 ✓")
print()
print(f"========== MÉTRICAS ADICIONAIS ==========")
print(f"Precisão (Precision): {precisao:.4f} ({precisao*100:.2f}%)")
print(f"Recall (Revocação):   {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:             {f1_score:.4f} ({f1_score*100:.2f}%)")