import numpy as np

class RedeNeural:
    """
    MLP com:
      - camada de entrada (n_entradas)
      - 1 camada oculta (n_ocultos)
      - 1 neurônio de saída

    Implementação no estilo do livro Deep Learning Book,
    Data Science Academy(https://www.datascienceacademy.com.br/),
    com backpropagation e mini-batch SGD.
    """

    def __init__(self, n_entradas: int, n_ocultos: int, taxa_aprendizado: float = 0.1):
        self.sizes = [n_entradas, n_ocultos, 1]
        self.num_layers = len(self.sizes)
        self.taxa_aprendizado = taxa_aprendizado

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]

        self.weights = [
            np.random.randn(y, x) * 0.1
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]


    @staticmethod
    def sigmoide(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoide_prime(z):
        s = 1.0 / (1.0 + np.exp(-z))
        return s * (1.0 - s)

    @staticmethod
    def cost_derivative(output_activations, y):
        # derivada do custo quadrático: 1/2 ||a - y||^2  -> (a - y)
        return output_activations - y

    # ==================== Feedforward ====================

    def feedforward(self, a):
        """Propaga a entrada 'a' pela rede inteira."""
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoide(np.dot(w, a) + b)
        return a


    def backprop(self, x, y):
        """
        Retorna uma tupla (nabla_b, nabla_w) representando
        o gradiente do custo C_x em relação a cada bias e peso.
        nabla_b e nabla_w têm o mesmo formato que self.biases e self.weights.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]  # ativações de cada camada
        zs = []            # valores z = w.a + b em cada camada

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoide(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * self.sigmoide_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoide_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w


    def update_mini_batch(self, mini_batch, eta):
        """
        Atualiza pesos e biases aplicando o gradiente médio de um mini-batch.
        mini_batch: lista de tuplas (x, y)
        eta: taxa de aprendizado
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        m = len(mini_batch)
        self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]


    def treinar(self, X: np.ndarray, Y: np.ndarray, epocas: int = 1000, mini_batch_size: int = 16):
        """
        X: matriz (N, n_entradas)
        Y: vetor (N,) com rótulos 0/1
        Retorna lista com erro médio (MSE) por época.
        """
        N = X.shape[0]
        dados = [
            (X[i].reshape(-1, 1), np.array([[Y[i]]], dtype=float))
            for i in range(N)
        ]

        historico_erro = []

        for epoca in range(1, epocas + 1):
            np.random.shuffle(dados)

            mini_batches = [
                dados[k:k + mini_batch_size]
                for k in range(0, N, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, self.taxa_aprendizado)

            erro_total = 0.0
            for x, y in dados:
                y_pred = self.feedforward(x)
                erro_total += 0.5 * np.linalg.norm(y_pred - y) ** 2

            erro_medio = erro_total / N
            historico_erro.append(erro_medio)

            if epoca == 1 or epoca == epocas or epoca % 10 == 0:
                print(f"Época {epoca}/{epocas} - Erro médio: {erro_medio:.6f}")

        return historico_erro


    def prever_proba(self, x: np.ndarray) -> float:
        """Retorna a probabilidade (saída da sigmóide)."""
        a = self.feedforward(x.reshape(-1, 1))
        return float(a[0, 0])

    def prever_classe(self, x: np.ndarray, limiar: float = 0.5) -> int:
        """Converte a probabilidade em classe 0/1 usando um limiar."""
        proba = self.prever_proba(x)
        return 1 if proba >= limiar else 0