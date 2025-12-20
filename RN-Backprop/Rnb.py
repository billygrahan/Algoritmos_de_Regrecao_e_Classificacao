import numpy as np

class RedeNeural:
    """
    MLP bem simples:
      - n_entradas
      - 1 camada oculta com n_ocultos neurônios (sigmóide)
      - 1 neurônio de saída (sigmóide)
    """

    def __init__(self, n_entradas: int, n_ocultos: int, taxa_aprendizado: float = 0.1):
        self.n_entradas = n_entradas
        self.n_ocultos = n_ocultos
        self.n_saidas = 1
        self.taxa_aprendizado = taxa_aprendizado

        # pesos_entrada_oculta[i, j] : peso da entrada j para o neurônio oculto i
        # shape: (n_ocultos, n_entradas)
        self.pesos_entrada_oculta = np.random.uniform(
            -0.5, 0.5, size=(self.n_ocultos, self.n_entradas)
        )

        # bias_oculta[i]
        self.bias_oculta = np.random.uniform(-0.5, 0.5, size=(self.n_ocultos,))

        # pesos_oculta_saida[i] : peso do neurônio oculto i para o neurônio de saída
        # shape: (n_ocultos,)
        self.pesos_oculta_saida = np.random.uniform(-0.5, 0.5, size=(self.n_ocultos,))

        # bias_saida (escalar)
        self.bias_saida = np.random.uniform(-0.5, 0.5)

        # ativacões (só para armazenar o último forward)
        self.ativ_oculta = np.zeros(self.n_ocultos)
        self.ativ_saida = 0.0

    # --- Funções de ativação ---

    @staticmethod
    def sigmoide(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def derivada_sigmoide(y):
        # aqui y já é sigmoide(x)
        return y * (1.0 - y)

    # --- Forward ---

    def forward(self, x: np.ndarray) -> float:
        """
        x: vetor de entrada com shape (n_entradas,)
        retorna: saída escalar (entre 0 e 1)
        """
        # camada oculta: soma ponderada + bias, depois sigmóide
        # z_oculta[i] = sum_j (w[i, j] * x[j]) + b[i]
        z_oculta = self.pesos_entrada_oculta @ x + self.bias_oculta
        self.ativ_oculta = self.sigmoide(z_oculta)

        # camada de saída (1 neurônio)
        # z_saida = sum_i (w_saida[i] * ativ_oculta[i]) + bias_saida
        z_saida = self.pesos_oculta_saida @ self.ativ_oculta + self.bias_saida
        self.ativ_saida = self.sigmoide(z_saida)

        return float(self.ativ_saida)

    # --- Backpropagation para 1 exemplo ---

    def backpropagation(self, x: np.ndarray, alvo: float):
        """
        Executa forward + backpropagation para UM exemplo (x, alvo)
        Atualiza os pesos in-place (gradient descent).
        """
        # 1) FORWARD
        saida = self.forward(x)

        # 2) Erro na saída (MSE: 0.5*(saida - alvo)^2 -> derivada: (saida - alvo))
        erro_saida = saida - alvo
        delta_saida = erro_saida * self.derivada_sigmoide(saida)

        # 3) Gradiente para pesos oculta->saida
        grad_pesos_oculta_saida = delta_saida * self.ativ_oculta  # shape: (n_ocultos,)
        self.pesos_oculta_saida -= self.taxa_aprendizado * grad_pesos_oculta_saida

        # bias da saída
        self.bias_saida -= self.taxa_aprendizado * delta_saida

        # 4) Deltas da camada oculta
        # erro propagado para cada neurônio oculto i
        erro_oculta = delta_saida * self.pesos_oculta_saida  # shape: (n_ocultos,)
        delta_oculta = erro_oculta * self.derivada_sigmoide(self.ativ_oculta)

        # 5) Gradiente para pesos entrada->oculta
        # para cada neurônio oculto i e entrada j:
        # grad_w[i, j] = delta_oculta[i] * x[j]
        grad_pesos_entrada_oculta = delta_oculta[:, np.newaxis] * x[np.newaxis, :]
        self.pesos_entrada_oculta -= self.taxa_aprendizado * grad_pesos_entrada_oculta

        # bias da camada oculta
        self.bias_oculta -= self.taxa_aprendizado * delta_oculta

        # retorna o erro (MSE) só para fins de monitoramento
        return 0.5 * (erro_saida ** 2)

    # --- Treino em lote (vários exemplos) ---

    def treinar(self, X: np.ndarray, Y: np.ndarray, epocas: int = 1000):
        """
        X: matriz (N, n_entradas)
        Y: vetor (N,) com alvos (0 ou 1, por exemplo)
        Retorna uma lista com o erro médio em cada época.
        """
        N = X.shape[0]
        historico_erro = []

        for epoca in range(1, epocas + 1):
            erro_total = 0.0
            for i in range(N):
                x = X[i]
                y = Y[i]
                erro_total += self.backpropagation(x, y)

            erro_medio = erro_total / N
            historico_erro.append(erro_medio)

            # prints de acompanhamento (a cada 10 épocas, e nas bordas)
            if epoca == 1 or epoca == epocas or epoca % 10 == 0:
                print(f"Época {epoca}/{epocas} - Erro médio: {erro_medio:.6f}")

        return historico_erro

    # --- Predição ---

    def prever_proba(self, x: np.ndarray) -> float:
        """Retorna a probabilidade (saída da sigmóide)."""
        return self.forward(x)

    def prever_classe(self, x: np.ndarray, limiar: float = 0.5) -> int:
        """Converte a probabilidade em classe 0/1 usando um limiar."""
        proba = self.prever_proba(x)
        return 1 if proba >= limiar else 0


# Exemplo mínimo de uso
if __name__ == "__main__":
    # Exemplo: porta lógica AND com 2 entradas
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    Y = np.array([0.0, 0.0, 0.0, 1.0])

    rede = RedeNeural(n_entradas=2, n_ocultos=3, taxa_aprendizado=0.5)
    rede.treinar(X, Y, epocas=5000)

    print("\nTestes:")
    for x, y in zip(X, Y):
        p = rede.prever_proba(x)
        c = rede.prever_classe(x)
        print(f"Entrada: {x}, alvo: {y}, proba: {p:.4f}, classe: {c}")