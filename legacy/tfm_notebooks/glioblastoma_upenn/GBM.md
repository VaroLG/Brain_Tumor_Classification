Resumen del Proyecto
Este proyecto implementa scripts en Python para la organización y división de conjuntos de datos de imágenes médicas relacionadas con el Glioblastoma Multiforme (GBM). El objetivo es preparar los datos para entrenar un modelo de Red Neuronal Convolucional (CNN) para la clasificación automática de tumores según diferentes marcadores de pronóstico y genéticos.

El código realiza las siguientes tareas de preprocesamiento de datos y clasificación:

Clasificación por Supervivencia Global (SG): Alta (SG alta) o Baja (SG baja).

Clasificación por Estado de Metilación de MGMT: Metilado (M) o No Metilado/No Clasificado (U).

Clasificación por Estado de Mutación de IDH1: Mutado (M) o Tipo Salvaje (W).

🚀 Estructura del Directorio (Entrada)
El script espera que los datos de entrada estén organizados en carpetas etiquetadas dentro de un directorio base.

Ejemplo para la Clasificación de Supervivencia Global (SG)
Tu directorio base debe contener las carpetas de etiquetas con las imágenes dentro:

directorio_base/
├── SG alta/
│   ├── imagen_001.png
│   └── ...
└── SG baja/
    ├── imagen_010.png
    └── ...
Nota: Estructuras similares son necesarias para las clasificaciones de MGMT (carpetas M y U) e IDH1 (carpetas W y M).

🛠️ Requisitos
Asegúrate de tener instalado Python 3 y las siguientes librerías:

os

random

shutil

💻 Uso
El notebook incluye secciones de código para la división de datos para las tres tareas de clasificación (SG, MGMT, IDH1).

1. Configurar Rutas
Antes de ejecutar cualquier celda de división de datos, debes editar la variable base_dir para que apunte a la ruta correcta de las imágenes de entrada:

Python

base_dir = r"ruta_absoluta_a_tus_imagenes"
2. División de Datos
El script realiza una división estratificada y aleatoria de las imágenes en los siguientes conjuntos, definidos por el diccionario data_split = {"train": 0.6, "val": 0.15, "test": 0.25}:

Entrenamiento (train): 60% de las imágenes para ajustar los pesos del modelo.

Validación (val): 15% de las imágenes para ajustar los hiperparámetros del modelo.

Prueba (test): 25% de las imágenes para la evaluación final del rendimiento del modelo.

Ejecuta la sección de código correspondiente a la clasificación que deseas procesar (SG, MGMT o IDH1).

3. Estructura del Directorio (Salida)
Tras la ejecución, el script crea carpetas de subconjuntos (train, val, test) dentro de la ruta base_dir, organizando las imágenes por conjunto y por etiqueta:

directorio_base/
├── SG alta/       # Carpeta original
├── SG baja/       # Carpeta original
├── train/
│   ├── SG alta/
│   └── SG baja/
├── val/
│   ├── SG alta/
│   └── SG baja/
└── test/
    ├── SG alta/
    └── SG baja/
💡 Próximos Pasos
Una vez que los datos estén divididos y organizados, se puede proceder con el entrenamiento del modelo de CNN:

Definir y construir la arquitectura del modelo de Red Neuronal Convolucional (CNN).

Entrenar el modelo utilizando el conjunto de train.

Evaluar el rendimiento final del modelo utilizando el conjunto de test.


Una vez clasificadas en carpetas en función del tipo de contraste y de corte, para este modelo de clasificación se escogió un unico contraste y corte, debido al gran número de imágenes generadas a partir del dataset original.

