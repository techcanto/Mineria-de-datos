import math as m
import numpy as np
import pandas as pd
import random as r

def euclidean_distance(point1, point2):

    p1 = np.asarray(point1, dtype=float)
    p2 = np.asarray(point2, dtype=float)
    
    if p1.shape != p2.shape:
        raise ValueError("Los puntos deben de tener la misma dimensión")
    
    return float(np.sqrt(np.sum((p1 - p2) ** 2)))



def create_kmeans(data, n_clusters=8, max_iter=1000, tol=0.0001, randomstate=None):
    
    # Convertir a numpy
    if hasattr(data, "to_numpy"):          # DataFrame
        X = data.to_numpy(dtype=float) # Si es de pandas lo pasamos a numpy
    else:
        X = np.asarray(data, dtype=float) # Cualquier otra cosa la convierte a numpy


    # Manejo de errores de captura de datos
    if n_clusters > len(X):
        raise ValueError("n_clusters no puede ser mayor que el número de puntos")
    if n_clusters < 1:
        raise ValueError("n_clusters debe ser al menos 1")


    
    rng = np.random.default_rng(randomstate) # Generador de números pseudo aleatorios de numpy
    
    indices = rng.choice(len(X), size=n_clusters, replace=False) # Arreglo de numpy con un numero n_clusters de datos y que cada uno puede tomar un valor en el rango entre 0 y len(X)
    centroides = X[indices].copy() # Matriz con solo las filas que curos indices estan en indices
    
    return { # Diccionario que actuca como objeto donde cada clave es un atributo
        "data": X,
        "n_clusters": int(n_clusters),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "randomstate": randomstate,
        "centroides": centroides,
        "etiquetas": None,
        "inercia_historial": [],
        "n_iter": 0,
        "converged": False,
    }








def kmeans_fit(kmeans, verbose=True):

    # Sacar los datos del diccionario
    X = kmeans["data"]
    k = kmeans["n_clusters"]
    centroides = kmeans["centroides"]
    limite = kmeans["max_iter"]

    # for para limite de convergencia
    for it in range(limite):
        
        # Paso 1: asignar cada punto al centroide más cercano
        etiquetas = np.zeros(len(X), dtype=int)                               # guarda el cluster asignado a cada punto
        for i, punto in enumerate(X):                                         # recorre cada fila con su indice
            distancias = [euclidean_distance(punto, c) for c in centroides]   # lista de k distancias una por centroide
            etiquetas[i] = int(np.argmin(distancias))                         # indice del menor valor (centroide más cercano)

        
        # Paso 2: recalcular centroides como la media de su cluster
        nuevos_centroides = centroides.copy()                                 # copia para no modificar el original al iterar
        for j in range(k):
            miembros = X[etiquetas == j]                                      # filas de X donde etiqueta vale j (matriz de los miembros del cluster j)
            if len(miembros) > 0:                                             
                nuevos_centroides[j] = miembros.mean(axis=0)                  # vector con las medias por columnas de longitud n_features 
            else:
                # Cluster vacío: reasignamos a un punto aleatorio
                if kmeans["randomstate"] is not None:
                    rng = np.random.default_rng(kmeans["randomstate"] + it)
                else:
                    rng = np.random.default_rng()
                nuevos_centroides[j] = X[rng.integers(len(X))]

        
        # Paso 3: comprobar convergencia
        desplazamiento = np.sqrt(np.sum( (nuevos_centroides - centroides) ** 2)) # norma de Frobenius
        centroides = nuevos_centroides
        kmeans["n_iter"] = it + 1                                              # registra las iteraciones que llevamos
        
        if verbose:
            print(f"Iteración {it + 1}: desplazamiento centroides = {desplazamiento:.6f}")

        # Criterio de paro por tolerancia
        if desplazamiento < kmeans["tol"]:                                      # Si los centroides se mueven menos que tol consideramos que ya se estabilizaron y rompemos el bucle
            kmeans["converged"] = True                                          # Marcamos como que ya convergio
            if verbose:
                print("Convergió.")
            break

    # Guardar los resultados en el diccionario
    kmeans["centroides"] = centroides
    kmeans["etiquetas"] = etiquetas
    return kmeans
