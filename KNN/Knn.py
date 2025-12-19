import math

def distancia_euclidiana(ponto1, ponto2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(ponto1, ponto2)))

# def mapeiamento_pontos

def knn(treino, labels, ponto_teste, k):
    distancias = []
    for ponto in treino:
        distancias.append(distancia_euclidiana(ponto_teste, ponto))

    indice_k_vizinhos = sorted(range(len(distancias)), key=lambda i: distancias[i])[:k]
    
    vizinhos_labels = []
    for i in indice_k_vizinhos:
        vizinhos_labels.append(labels[i])

    contagem = {}
    for label in vizinhos_labels:
        if label in contagem:
            contagem[label] += 1
        else:
            contagem[label] = 1
    
    classe_predita = max(contagem, key=contagem.get)
    return classe_predita