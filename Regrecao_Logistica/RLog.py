import math
import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)  # Evita overflow
    return 1 / (1 + np.exp(-z))

def custo_logistico(x, y, w, b):
    m = len(y)
    z = np.dot(x, w) + b
    y_hat = sigmoid(z)

    epsilon = 1e-15

    return -(1/m) * sum(
        y * np.log(y_hat + epsilon) +
        (1 - y) * np.log(1 - y_hat + epsilon)
    )


def treinar(x, y, tx_aprendizado, ciclos):
    m, n = x.shape 
    w = np.zeros(n) 
    b = 0.0  

    custo_anterior = float('inf')

    for epoch in range(ciclos):  
        z = np.dot(x, w) + b  
        y_pred = sigmoid(z)  

        dw = (1/m) * np.dot(x.T, (y_pred - y))  # Derivada do custo em relação aos pesos w
        db = (1/m) * sum(y_pred - y)  # Derivada do custo em relação ao bias b (sum built-in)

        w -= tx_aprendizado * dw  
        b -= tx_aprendizado * db 


        custo_atual = custo_logistico(x, y, w, b)
    
        if abs(custo_anterior - custo_atual) < 0.0001:
            break
        
        custo_anterior = custo_atual

    return w, b 


def prever(amostra, w, b):
    z = 0
    for i in range(len(amostra)):
        z += w[i] * amostra[i]
    z += b
    
    probabilidade = sigmoid(z)
    
    if probabilidade >= 0.5:
        classe = 1
    else:
        classe = 0
    
    return classe, probabilidade