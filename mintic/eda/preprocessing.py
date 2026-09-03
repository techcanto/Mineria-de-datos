# Código para poner las funciones de preprocesamiento

import pandas as pd

# data:         Dataframe de pandas
# columns:      lista de strings de nombres de las columnas a revisar (pasar solo columnas numericas)

def impute_missing(data, strategy="mean", columns=None):
        booleanosData = [_ for _ in data.isna().any()]
        if any(booleanosData):
                if columns == None:
                        columnas = list(data.columns)
                else:
                        columnas = columns

                
                resultado = data.copy()
                numFilas = len(resultado[columnas].values)

                if strategy == "mean":
                        for columna in columnas:
                                #suma = sum(resultado[columna].values)
                                suma = resultado[columna].sum()
                                cantidad = resultado[columna].count()
                                media = suma / cantidad
                                resultado[columna] = resultado[columna].fillna(media)
                                #resultado[columna] = resultado[columna].fillna(resultado[columna].mean())

                elif strategy == "median":
                        for columna in columnas:
                                resultado_ordenado = resultado.sort_values(by=f"{columna}", ascending=True)
                                valor_medio = resultado_ordenado.iloc[numFilas // 2][f"{columna}"]
                                resultado[columna] = resultado[columna].fillna(valor_medio)
                                #print("Se aplicó la mediana a las columnas faltantes")

                elif strategy == "mode":
                        for columna in columnas:
                                conteos = resultado[f"{columna}"].value_counts()
                                moda = conteos.idxmax()
                                resultado[columna] = resultado[columna].fillna(moda)
                                #print("Se aplicó la moda a las columnas faltantes")
                return resultado
        else:
                print("No hay valores NAN")





#------------------------------------------------------------------------------
# method: es el método estadístico a utilizar ('iqr' o 'zscore')
# threshold: es el factor de escala o umbral para definir un valor atípico (por ejemplo 1.5 o 3.0)

# z = (x - media) / desvest

# IQR = Q3 − Q1 (25% - 75%)
# Outliers con IQR:
# - Límite inferior: Q1 - threshold*IQR
# - Límite superior: Q3 + threshold*IQR

def detect_outliers(data, method='iqr', threshold=1.5):
    
    resultado = data.copy()
    data_numericos = resultado.select_dtypes(include="number")
    columnas = list(data_numericos.columns)
    
    if method == "iqr":
        for columna in columnas:

            cuartiles = data_numericos[columna].quantile([0.25, 0.75])
            Q1, Q3 = cuartiles[0.25], cuartiles[0.75]
            IQR = Q3 - Q1

            limite_inferior = Q1 - threshold * IQR
            limite_superior = Q3 + threshold * IQR

            outliers = (data_numericos[columna] < limite_inferior) | (data_numericos[columna] > limite_superior)

            resultado[columna] = outliers



    elif method == "zscore":
        for columna in columnas:

            suma = data_numericos[columna].sum()
            cantidad = data_numericos[columna].count()
            media = suma / cantidad
            desvest = data_numericos[columna].std()

            zscore = (data_numericos[columna] - media) / desvest
            outlier = zscore.abs() > threshold

            resultado[columna] = outlier
              


    # Regresa Un DataFrame de booleanos True si ese dato es un outlier, False si no es un outlier
    return resultado[columnas]







#-----------------------------------------------------------------------------
