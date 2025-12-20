import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

sys.path.append(os.path.join(project_root, 'Bases'))
sys.path.append(os.path.join(project_root, 'KNN'))

from Base_breast_cancer import data
from Knn import knn

X = data.data.values.tolist()
y = data.target.values.tolist()

dados_combinados = list(zip(X, y))
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
k = 15
print(f"Testando KNN com k={k}...")

acertos = 0
total_testes = len(X_teste)

# Variáveis para a matriz de confusão
TP = 0  # True Positive: previu 1 e era 1
TN = 0  # True Negative: previu 0 e era 0
FP = 0  # False Positive: previu 1 mas era 0
FN = 0  # False Negative: previu 0 mas era 1

for i, ponto_teste in enumerate(X_teste):
    predicao = knn(X_treino, y_treino, ponto_teste, k)
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

# ========== VISUALIZAÇÕES ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'KNN (k={k}) - Análise de Desempenho', fontsize=16, fontweight='bold')

# Gráfico 1: Matriz de Confusão (Heatmap)
matriz_confusao = np.array([[TN, FP], [FN, TP]])
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[0],
            xticklabels=['Previsto 0', 'Previsto 1'],
            yticklabels=['Real 0', 'Real 1'])
axes[0].set_title('Matriz de Confusão', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Valor Real', fontsize=10)
axes[0].set_xlabel('Valor Previsto', fontsize=10)

# Gráfico 2: Métricas em Barras
metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
valores = [acuracia/100, precisao, recall, f1_score]
cores = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

bars = axes[1].bar(metricas, valores, color=cores, alpha=0.7, edgecolor='black')
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel('Score', fontsize=10)
axes[1].set_title('Métricas de Desempenho', fontsize=12, fontweight='bold')
axes[1].axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
axes[1].grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bar, valor in zip(bars, valores):
    altura = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., altura + 0.02,
                f'{valor:.4f}\n({valor*100:.2f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
# plt.savefig('resultados_knn.png', dpi=300, bbox_inches='tight')
# print("\n✓ Gráfico salvo como 'resultados_knn.png'")
plt.show()
