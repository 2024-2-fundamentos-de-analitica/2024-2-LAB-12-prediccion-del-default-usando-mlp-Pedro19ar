# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
import gzip
import pickle
import json
import os
from glob import glob
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Función para cargar los datos
def cargar_datos():
    df_train = pd.read_csv("./files/input/train_data.csv.zip", compression="zip")
    df_test = pd.read_csv("./files/input/test_data.csv.zip", compression="zip")
    return df_train, df_test

# Función para limpiar los datos
def limpiar_datos(df):
    df = df.rename(columns={'default payment next month': 'default'}).drop(columns=["ID"])
    df = df[(df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
    return df.dropna()

# Separar variables predictoras y objetivo
def dividir_datos(df):
    return df.drop(columns=["default"]), df["default"]

# Definir pipeline de preprocesamiento y modelo
def construir_pipeline(x_train):
    categorias = ["SEX", "EDUCATION", "MARRIAGE"]
    numericas = [col for col in x_train.columns if col not in categorias]

    preprocesador = ColumnTransformer([
        ('cat', OneHotEncoder(), categorias),
        ('scaler', StandardScaler(), numericas),
    ])

    return Pipeline([
        ("preprocesador", preprocesador),
        ("seleccion_caracteristicas", SelectKBest(score_func=f_classif)),
        ("pca", PCA()),
        ("clasificador", MLPClassifier(max_iter=15000, random_state=21))
    ])

# Configurar optimización de hiperparámetros
def configurar_estimador(pipeline):
    parametros = {
        "pca__n_components": [None],
        "seleccion_caracteristicas__k": [20],
        "clasificador__hidden_layer_sizes": [(50, 30, 40, 60)],
        "clasificador__alpha": [0.26],
        "clasificador__learning_rate_init": [0.001],
    }
    return GridSearchCV(
        estimator=pipeline, param_grid=parametros, cv=10,
        scoring='balanced_accuracy', n_jobs=-1, refit=True
    )

# Guardar el modelo
def guardar_modelo(ruta, modelo):
    os.makedirs("files/models", exist_ok=True)
    with gzip.open(ruta, "wb") as f:
        pickle.dump(modelo, f)

# Calcular métricas de evaluación
def calcular_metricas(tipo, y_real, y_predicho):
    return {
        "type": "metrics", "dataset": tipo,
        "precision": precision_score(y_real, y_predicho, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_predicho),
        "recall": recall_score(y_real, y_predicho, zero_division=0),
        "f1_score": f1_score(y_real, y_predicho, zero_division=0)
    }

# Calcular matriz de confusión
def calcular_matriz_confusion(tipo, y_real, y_predicho):
    cm = confusion_matrix(y_real, y_predicho)
    return {
        "type": "cm_matrix", "dataset": tipo,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])}
    }

# Ejecutar el flujo de trabajo
def ejecutar_proceso():
    df_train, df_test = cargar_datos()
    df_train, df_test = limpiar_datos(df_train), limpiar_datos(df_test)
    x_train, y_train = dividir_datos(df_train)
    x_test, y_test = dividir_datos(df_test)
    
    modelo_pipeline = construir_pipeline(x_train)
    modelo = configurar_estimador(modelo_pipeline)
    modelo.fit(x_train, y_train)
    
    guardar_modelo("files/models/model.pkl.gz", modelo)
    
    y_pred_train = modelo.predict(x_train)
    y_pred_test = modelo.predict(x_test)
    
    metricas_train = calcular_metricas("train", y_train, y_pred_train)
    metricas_test = calcular_metricas("test", y_test, y_pred_test)
    matriz_train = calcular_matriz_confusion("train", y_train, y_pred_train)
    matriz_test = calcular_matriz_confusion("test", y_test, y_pred_test)
    
    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        for entrada in [metricas_train, metricas_test, matriz_train, matriz_test]:
            file.write(json.dumps(entrada) + "\n")

if __name__ == "__main__":
    ejecutar_proceso()
