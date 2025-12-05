import math

def distancia_euclidiana(ponto1, ponto2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(ponto1, ponto2)))

def knn(treino, labels, ponto_teste, k):
    distancias = [distancia_euclidiana(ponto_teste, ponto) for ponto in treino]
    
    indices_vizinhos = sorted(range(len(distancias)), key=lambda i: distancias[i])[:k]
    
    vizinhos_labels = [labels[i] for i in indices_vizinhos]
    
    classes = list(set(vizinhos_labels))
    vizinhos_labels = [classes.index(label) for label in vizinhos_labels]
    
    contagem_labels = [0] * len(classes)
    for label in vizinhos_labels:
        contagem_labels[label] += 1
    
    classe_predita = classes[contagem_labels.index(max(contagem_labels))]
    return classe_predita