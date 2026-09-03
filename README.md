# Mineria-de-datos

Curso de Minería de Datos

**Antonio S**
**ENES Morelia, UNAM**

## Sobre el proyecto

`Mineria-de-datos` es el repositorio en el que construiré mi propia librería de Minería de Datos. El objetivo es comprender el funcionamiento interno de los algoritmos para la signatura de Míneria de datos (aprendizaje automático no supervisado), no únicamente usarlos.

## Reglas del desarrollo

- Toda la lógica de los algoritmos debe implementarse manualmente por mi dentro del subpaquete correspondiente en `Mineria-de-datos/`.
- La única librería numérica permitida para los cálculos del algoritmo es **NumPy**. No está permitido usar `scikit-learn`, `scipy` u otras librerías que ya implementen el algoritmo asignado.
- `pandas` se permite únicamente para la carga y manipulación inicial de datos (lectura de CSV, por ejemplo).
- `matplotlib` se permite para la visualización de resultados.

## Estructura del repositorio

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── data/
│   └── sample_dataset.csv
└── mintic/
    ├── __init__.py
    ├── eda/
    │   └── __init__.py
    ├── ensemble/
    │   └── __init__.py
    ├── kmeans/
    │   └── __init__.py
    ├── dbscan/
    │   └── __init__.py
    ├── apriori/
    │   └── __init__.py
    └── pca/
        └── __init__.py
```

Cada subcarpeta dentro de `Mineria-de-datos/` es un subpaquete de Python correspondiente a un reto del curso. Mi papel es desarrollar mi implementación dentro del subpaquete que me sea asignado.

## Clonar el repositorio

Ejecuta:

```bash
git clone https://github.com/techcanto/Mineria-de-datos.git
cd Mineria-de-datos
```

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

`main.py` carga el dataset de prueba en `data/sample_dataset.csv` y sirve como punto de partida para importar y probar el subpaquete que estés desarrollando.
