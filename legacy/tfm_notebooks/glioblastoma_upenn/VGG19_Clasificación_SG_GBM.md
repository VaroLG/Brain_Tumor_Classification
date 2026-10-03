Clasificación de Glioblastoma Multiforme (GBM) por Supervivencia Global (SG) usando CNN (VGG19)

📄 Resumen del Proyecto
Este proyecto se centra en la aplicación de una Red Neuronal Convolucional (CNN) preentrenada, específicamente VGG19, mediante la técnica de Transfer Learning, para clasificar imágenes de Glioblastoma Multiforme (GBM) basándose en la Supervivencia Global (SG) del paciente.

El objetivo es entrenar un modelo para distinguir entre:

SG alta: Supervivencia Global alta.

SG baja: Supervivencia Global baja.

El código aborda la carga de datos, el aumento de datos (data augmentation), la construcción del modelo VGG19 (ajustando la capa densa de salida), y el entrenamiento del modelo.

🛠️ Requisitos e Instalación
Asegúrate de tener un entorno Python 3 con las siguientes librerías instaladas. Se recomienda usar pip para su instalación:

Bash

pip install tensorflow
pip install keras
pip install numpy
pip install matplotlib
Requisitos Adicionales
El entrenamiento del modelo VGG19 es intensivo y se beneficia enormemente de una GPU con soporte para CUDA.

📂 Estructura del Directorio de Datos
Este proyecto asume que los datos de las imágenes ya han sido divididos en conjuntos de entrenamiento, validación y prueba, organizados por las etiquetas (SG alta y SG baja).

La estructura esperada del directorio base (dir) es la siguiente:

dir/
├── train/
│   ├── SG alta/      # Imágenes para entrenamiento
│   └── SG baja/
├── val/
│   ├── SG alta/      # Imágenes para validación
│   └── SG baja/
└── test/
    ├── SG alta/      # Imágenes para prueba
    └── SG baja/
Configuración de Rutas
Asegúrate de configurar la variable dir en tu notebook para que apunte a la ruta base de tus datos:

Python

dir = r"C:\\ruta\\a\\tu\\directorio\\de\\imagenes\\de\\Glioma_clasificación"

Configuración del Modelo y Entrenamiento
1. Preprocesamiento y Data Augmentation
Se utiliza ImageDataGenerator de Keras para:

Normalizar los valores de píxeles (reescalando a [0,1]).

Aumentar los datos en el conjunto de entrenamiento mediante rotaciones, cambios de ancho/alto, cizallamiento y zoom, para mejorar la robustez del modelo.

Cargar las imágenes directamente desde la estructura de directorios.

Las imágenes se configuran con un tamaño de 224×224 píxeles, adecuado para VGG19.

2. Construcción del Modelo
El modelo se construye mediante Transfer Learning:

Se carga la arquitectura base de VGG19, preentrenada con el dataset ImageNet, y se congelan sus capas convolucionales (base_model.trainable = False).

Se añade una capa de aplanamiento (Flatten).

Se añade una capa densa (Dense) con una función de activación ReLU.

Se añade la capa de salida (Dense) con 2 unidades (una por clase: SG alta/baja) y activación Softmax.

3. Compilación y Entrenamiento
El modelo se compila con:

Optimizador: Adam.

Función de Pérdida: categorical_crossentropy (apropiada para clasificación multiclase).

Métrica: accuracy.

El entrenamiento se realiza durante un número definido de épocas, utilizando fit_generator para iterar sobre los datos generados.

Evaluación de Resultados
Una vez completado el entrenamiento, el modelo debe ser evaluado en el conjunto de prueba (test_generator) para determinar su rendimiento en datos no vistos. Los resultados de pérdida y precisión se reportarán para validar la capacidad de clasificación del modelo VGG19.
Contribuciones
Siéntete libre de clonar este repositorio, experimentar con diferentes arquitecturas de CNN (como ResNet o EfficientNet), ajustar los hiperparámetros, o sugerir mejoras en el preprocesamiento de imágenes.
