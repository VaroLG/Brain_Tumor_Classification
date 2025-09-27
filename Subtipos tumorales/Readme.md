# Clasificación de Imágenes con EfficientNetB1

Este repositorio contiene un modelo de **clasificación de imágenes** basado en **EfficientNetB1**, implementado con TensorFlow/Keras. El modelo se entrena en múltiples etapas (30 épocas) aplicando técnicas de regularización como **Dropout** y **Batch Normalization** para evitar sobreajuste.

## Contenido

- `Clasificación_EfficientNetB1-30 etapas-Dropout.ipynb`: Notebook principal con todo el flujo de trabajo y comentarios explicativos.
- Carpeta `data/`: Estructura de datos con las imágenes organizadas en subcarpetas por clase (no incluida en el repositorio).

## Requisitos

Antes de ejecutar el notebook, asegúrate de tener instaladas las siguientes dependencias:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Flujo de trabajo y código

El notebook está organizado en secciones comentadas para facilitar la comprensión:

1. **Importación de librerías**: se cargan las librerías necesarias para la manipulación de datos, visualización y construcción del modelo. Algunas importaciones redundantes han sido marcadas en comentarios.

2. **Carga de datos**: se utiliza `ImageDataGenerator` para leer las imágenes desde carpetas y aplicar aumentación de datos (rotaciones, flips, etc.), mejorando la robustez del modelo.

3. **Exploración y visualización**: con `matplotlib` y `seaborn` se visualizan ejemplos de imágenes y la distribución de clases, lo que ayuda a identificar posibles desbalances.

4. **Definición del modelo**:

   - Se carga `EfficientNetB1` preentrenado en ImageNet como base del modelo.
   - Se añaden capas densas adicionales con activación `relu`, junto con **Batch Normalization** y **Dropout** para mejorar la generalización y reducir el sobreajuste.
   - La capa final utiliza `softmax` para obtener probabilidades de cada clase.

5. **Entrenamiento**:

   - Se compila el modelo con `Adam` como optimizador y función de pérdida adecuada para clasificación multiclase.
   - Se monitoriza la pérdida y la exactitud durante el entrenamiento.
   - Se aplican callbacks como `EarlyStopping` y `ModelCheckpoint` (si están activados) para detener el entrenamiento a tiempo y guardar el mejor modelo.

6. **Evaluación**:

   - Se calculan métricas con `classification_report` (precisión, recall, F1-score).
   - Se construye una matriz de confusión con `sklearn.metrics` para visualizar los aciertos y errores en cada clase.

Cada celda del notebook incluye comentarios que explican la lógica del código, desde la preparación de los datos hasta la interpretación de resultados.

## Uso

Para ejecutar el proyecto:

```bash
jupyter notebook Clasificación_EfficientNetB1-30 etapas-Dropout.ipynb
```

## Notas sobre las librerías

- `sklearn.metrics` estaba importado dos veces → se dejó una sola importación.
- `Sequential`, `Conv2D`, `MaxPool2D`, `Flatten` podrían eliminarse si no se usan en el modelo final.
- `re` y `defaultdict` también podrían eliminarse si no se usan en la preparación de datos.

## Autor

Álvaro Linacero Gracia

