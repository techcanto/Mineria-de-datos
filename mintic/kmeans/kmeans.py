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



