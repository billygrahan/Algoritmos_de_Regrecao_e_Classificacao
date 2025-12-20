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

        self.pesos_entrada_oculta = np.random.uniform(
            -0.5, 0.5, size=(self.n_ocultos, self.n_entradas)
        )

        self.bias_oculta = np.random.uniform(-0.5, 0.5, size=(self.n_ocultos,))

        self.pesos_oculta_saida = np.random.uniform(-0.5, 0.5, size=(self.n_ocultos,))

        self.bias_saida = np.random.uniform(-0.5, 0.5)

        self.ativ_oculta = np.zeros(self.n_ocultos)
        self.ativ_saida = 0.0


    @staticmethod
    def sigmoide(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def derivada_sigmoide(y):
        return y * (1.0 - y)


    def forward(self, x: np.ndarray) -> float:
        """
        x: vetor de entrada com shape (n_entradas,)
        retorna: saída escalar (entre 0 e 1)
        """
        z_oculta = self.pesos_entrada_oculta @ x + self.bias_oculta
        self.ativ_oculta = self.sigmoide(z_oculta)

        z_saida = self.pesos_oculta_saida @ self.ativ_oculta + self.bias_saida
        self.ativ_saida = self.sigmoide(z_saida)

        return float(self.ativ_saida)


    def backpropagation(self, x: np.ndarray, alvo: float):
        """
        Executa forward + backpropagation para UM exemplo (x, alvo)
        Atualiza os pesos in-place (gradient descent).
        """
        saida = self.forward(x)

        erro_saida = saida - alvo
        delta_saida = erro_saida * self.derivada_sigmoide(saida)

        grad_pesos_oculta_saida = delta_saida * self.ativ_oculta  
        self.pesos_oculta_saida -= self.taxa_aprendizado * grad_pesos_oculta_saida

        self.bias_saida -= self.taxa_aprendizado * delta_saida

        erro_oculta = delta_saida * self.pesos_oculta_saida  
        delta_oculta = erro_oculta * self.derivada_sigmoide(self.ativ_oculta)

        grad_pesos_entrada_oculta = delta_oculta[:, np.newaxis] * x[np.newaxis, :]
        self.pesos_entrada_oculta -= self.taxa_aprendizado * grad_pesos_entrada_oculta

        self.bias_oculta -= self.taxa_aprendizado * delta_oculta

        return 0.5 * (erro_saida ** 2)


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

            if epoca == 1 or epoca == epocas or epoca % 10 == 0:
                print(f"Época {epoca}/{epocas} - Erro médio: {erro_medio:.6f}")

        return historico_erro


    def prever_proba(self, x: np.ndarray) -> float:
        """Retorna a probabilidade (saída da sigmóide)."""
        return self.forward(x)

    def prever_classe(self, x: np.ndarray, limiar: float = 0.5) -> int:
        """Converte a probabilidade em classe 0/1 usando um limiar."""
        proba = self.prever_proba(x)
        return 1 if proba >= limiar else 0
