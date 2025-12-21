import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../Bases')
sys.path.append('../RN-Backprop')

from Base_breast_cancer import data
from Rnb import RedeNeural

def get_plot_dir():
    """
    Retorna o caminho da pasta Plots/Backpropagation (cria se não existir).
    """
    base_dir = os.path.dirname(__file__)
    plot_dir = os.path.join(base_dir, '..', 'Plots', 'Backpropagation')
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def get_plot_dir():
    """
    Retorna o caminho da pasta Plots/Backpropagation (cria se não existir).
    """
    base_dir = os.path.dirname(__file__)
    plot_dir = os.path.join(base_dir, '..', 'Plots', 'Backpropagation')
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def carregar_dados():
    """
    Usa o objeto 'data' do sklearn (já carregado em Base_breast_cancer.py)
    e devolve X, y como numpy arrays normalizados.
    """
    X = data.data.to_numpy(dtype=float)
    y = data.target.to_numpy(dtype=float) 

    media = X.mean(axis=0)
    desvio = X.std(axis=0)
    desvio[desvio == 0] = 1.0
    X_norm = (X - media) / desvio

    return X_norm, y


def dividir_treino_teste(X, y, proporcao_treino=0.8):
    """
    Embaralha os dados e separa:
      - proporcao_treino (80%) para treino
      - o resto (20%) para teste
    """
    N = X.shape[0]
    idx = np.random.permutation(N)
    X = X[idx]
    y = y[idx]

    n_treino = int(proporcao_treino * N)
    X_treino = X[:n_treino]
    y_treino = y[:n_treino]
    X_teste = X[n_treino:]
    y_teste = y[n_treino:]
    return X_treino, y_treino, X_teste, y_teste


def avaliar_rede(rede: RedeNeural, X_teste, y_teste, limiar=0.5):
    plot_dir = get_plot_dir()

    N = X_teste.shape[0]
    y_pred = []
    for i in range(N):
        proba = rede.prever_proba(X_teste[i])
        classe = 1 if proba >= limiar else 0
        y_pred.append(classe)
    y_pred = np.array(y_pred, dtype=int)
    y_true = y_teste.astype(int)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    total = TP + TN + FP + FN

    acuracia = (TP + TN) / total if total > 0 else 0.0
    precisao = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) > 0 else 0.0


    print("\nMatriz de confusão:")
    print(f"TN: {TN}  FP: {FP}")
    print(f"FN: {FN}  TP: {TP}\n")

    print(f"Acurácia: {acuracia:.4f}")
    print(f"Precisão: {precisao:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"F1:       {f1:.4f}")

    # ---------- GRÁFICO: MATRIZ DE CONFUSÃO ----------
    cm = np.array([[TN, FP],
                   [FN, TP]])

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Negativa (0)", "Positiva (1)"])
    plt.yticks(tick_marks, ["Negativa (0)", "Positiva (1)"])

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("Classe real")
    plt.xlabel("Classe predita")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "matriz_confusao.png"), dpi=300)

    # ---------- GRÁFICO: MÉTRICAS (ACURÁCIA, PRECISÃO, RECALL, F1) ----------
    metricas = ["Acurácia", "Precisão", "Recall", "F1"]
    valores = [acuracia, precisao, recall, f1]

    plt.figure(figsize=(6, 4))
    plt.bar(metricas, valores, color=["C0", "C1", "C2", "C3"])
    plt.ylim(0, 1.05)
    plt.ylabel("Valor")
    plt.title("Métricas de Desempenho")
    for i, v in enumerate(valores):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "metricas_desempenho.png"), dpi=300)
    plt.show()


def main():
    X, y = carregar_dados()
    print("Formato de X:", X.shape)
    print("Formato de y:", y.shape)

    X_treino, y_treino, X_teste, y_teste = dividir_treino_teste(X, y, proporcao_treino=0.8)
    print("Treino:", X_treino.shape[0], "amostras")
    print("Teste :", X_teste.shape[0], "amostras")

    n_entradas = X_treino.shape[1]
    n_ocultos = 16
    rede = RedeNeural(n_entradas=n_entradas, n_ocultos=n_ocultos, taxa_aprendizado=0.01)

    historico_erro = rede.treinar(X_treino, y_treino, epocas=100)

    plot_dir = get_plot_dir()

    fig_erro = plt.figure()
    plt.plot(historico_erro)
    plt.xlabel("Época")
    plt.ylabel("Erro médio (MSE)")
    plt.title("Evolução do erro no treino")
    plt.grid(True)
    fig_erro.savefig(os.path.join(plot_dir, "erro_treino.png"), dpi=300)
    plt.show()

    avaliar_rede(rede, X_teste, y_teste, limiar=0.5)


if __name__ == "__main__":
    main()