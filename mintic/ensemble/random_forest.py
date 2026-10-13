import numpy as np

def bootstrap_sample(X, y, random_state=None): # Toma N muestras aleatorias con reemplazo de X e y

    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(y)
    indices = np.random.randint(0, n, size=n)  # N índices con reemplazo
    return X[indices], y[indices]



def _entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-12))

def _information_gain(X_col, y):
    total_entropy = _entropy(y)
    valores, counts = np.unique(X_col, return_counts=True)
    weighted = 0
    for v, c in zip(valores, counts):
        weighted += (c / len(y)) * _entropy(y[X_col == v])
    return total_entropy - weighted

def build_id3_tree(X, y, atributos=None):

    if len(np.unique(y)) == 1:
        return {"prediccion": y[0]}
    
    if atributos is None:
        atributos = list(range(X.shape[1]))
    
    if len(atributos) == 0:
        vals, counts = np.unique(y, return_counts=True)
        return {"prediccion": vals[np.argmax(counts)]}
    
    # Elegir mejor atributo por ganancia
    gains = [(a, _information_gain(X[:, a], y)) for a in atributos]
    best_attr = max(gains, key=lambda t: t[1])[0]
    
    tree = {"atributo": best_attr, "hijos": {}}
    vals, counts = np.unique(y, return_counts=True)
    tree["prediccion"] = vals[np.argmax(counts)]  # respaldo
    
    restantes = [a for a in atributos if a != best_attr]
    for v in np.unique(X[:, best_attr]):
        mask = X[:, best_attr] == v
        tree["hijos"][v] = build_id3_tree(X[mask], y[mask], restantes)
    
    return tree




def build_random_forest(X, y, n_trees=10, random_state=42): # Entrena n arboles cada uno sobre un bootstrap distinto

    forest = []
    for i in range(n_trees):
        seed = None if random_state is None else random_state + i
        X_s, y_s = bootstrap_sample(X, y, random_state=seed)
        tree = build_id3_tree(X_s, y_s)
        forest.append(tree)
    return forest





def _predict_tree(tree, x):
    if "hijos" not in tree:
        return tree["prediccion"]
    attr = tree["atributo"]
    valor = x[attr]
    if valor in tree["hijos"]:
        return _predict_tree(tree["hijos"][valor], x)
    return tree["prediccion"]  # respaldo

def predict_ensemble(trees, X_test): # Cada arbol vota y gana la clase mayoritaria
    predicciones = []
    for x in X_test:
        votos = {}
        for t in trees:
            p = _predict_tree(t, x)
            votos[p] = votos.get(p, 0) + 1
        # Desempate: primera clase alfabéticamente
        max_votos = max(votos.values())
        candidatos = sorted([k for k, v in votos.items() if v == max_votos])
        predicciones.append(candidatos[0])
    return np.array(predicciones)
