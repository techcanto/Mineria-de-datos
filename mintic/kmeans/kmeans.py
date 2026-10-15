import math as m

def euclidean_distance(point1:list, point2:list):
    # Vectores en forma de lista
    resultados = []
    
    if len(point1) != len(point2):
        raise ValueError("Los puntos deben de tener la misma dimensión")
        
    for i in range(len(point1)):
        resultados.append( (point1[i] - point2[i])**2 )
        
    return m.sqrt(sum(resultados))
