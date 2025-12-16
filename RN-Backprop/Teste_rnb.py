import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../Bases')
sys.path.append('../RN-Backprop')

# importa a base (já existe)
from Base_breast_cancer import data
# importa sua rede (arquivo Rede_neural.py está na mesma pasta deste script)
from Rnb import RedeNeural


def carregar_dados():
    """
    Usa o objeto 'data' do sklearn (já carregado em Base_breast_cancer.py)
    e devolve X, y como numpy arrays.
    """
    # data.data e data.target são DataFrame/Série do pandas
    X = data.data.to_numpy(dtype=float)    # shape: (N, n_features)
    y = data.target.to_numpy(dtype=float)  # shape: (N,)

    # Normalização simples (padronização): média 0, desvio 1
    media = X.mean(axis=0)
    desvio = X.std(axis=0)
    # evita divisão por zero
    desvio[desvio == 0] = 1.0
    X_norm = (X - media) / desvio

    return X_norm, y


def dividir_treino_teste(X, y, proporcao_treino=0.8):
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
    N = X_teste.shape[0]
    y_pred = []
    for i in range(N):
        proba = rede.prever_proba(X_teste[i])
        classe = 1 if proba >= limiar else 0
        y_pred.append(classe)
    y_pred = np.array(y_pred, dtype=int)
    y_true = y_teste.astype(int)

    # Matriz de confusão
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


def main():
    # 1) Carrega e normaliza os dados da pasta Bases
    X, y = carregar_dados()
    print("Formato de X:", X.shape)
    print("Formato de y:", y.shape)

    # 2) Divide em treino e teste
    X_treino, y_treino, X_teste, y_teste = dividir_treino_teste(X, y, proporcao_treino=0.8)
    print("Treino:", X_treino.shape[0], "amostras")
    print("Teste :", X_teste.shape[0], "amostras")

    # 3) Cria a rede
    n_entradas = X_treino.shape[1]
    n_ocultos = 16
    rede = RedeNeural(n_entradas=n_entradas, n_ocultos=n_ocultos, taxa_aprendizado=0.01)

    # 4) Treina (agora recebendo o histórico de erro)
    historico_erro = rede.treinar(X_treino, y_treino, epocas=100)

    # 4.1) Plota o erro por época
    plt.figure()
    plt.plot(historico_erro)
    plt.xlabel("Época")
    plt.ylabel("Erro médio (MSE)")
    plt.title("Evolução do erro no treino")
    plt.grid(True)
    plt.show()

    # 5) Avalia
    avaliar_rede(rede, X_teste, y_teste, limiar=0.5)


if __name__ == "__main__":
    main()