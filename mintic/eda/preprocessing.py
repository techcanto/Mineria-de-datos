# Código para poner las funciones de preprocesamiento

import pandas as pd

# data:Dataframe de pandas
# columns: lista de nombres de las columnas a revisar (pasar solo columnas numericas)

def impute_missing(data, strategy="mean", columns=None):
        booleanosData = [_ for _ in data.isna().any()]
        if any(booleanosData):
                #columnas = list(data.columns)
                resultado = data.copy()

                if strategy == "mean":
                        for columna in columns:
                                resultado[columna] = resultado[columna].fillna(resultado[columna].mean())
                                #print("Se aplicó la media a las columnas faltantes")
                elif strategy == "median":
                        for columna in columns:
                                resultado[columna] = resultado[columna].fillna(resultado[columna].median())
                                #print("Se aplicó la mediana a las columnas faltantes")
                elif strategy == "mode":
                        for columna in columns:
                                resultado[columna] = resultado[columna].fillna(resultado[columna].mode()[0])
                                #print("Se aplicó la moda a las columnas faltantes")
                return resultado
        else:
                print("No hay valores NAN")










#------------------------------------------------------------------------------
# method: es el método estadístico a utilizar ('iqr' o 'zscore')
# threshold: es el factor de escala o umbral para definir un valor atípico (por ejemplo 1.5 o 3.0)

def detect_outliers(data, method='iqr', threshold=1.5):
    pass # Regresa Un DataFrame de booleanos (True/False) indicando la ubicación de los outliers

