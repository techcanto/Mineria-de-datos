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
