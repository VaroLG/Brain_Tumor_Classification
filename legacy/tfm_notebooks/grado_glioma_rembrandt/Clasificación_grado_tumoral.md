# Clasificación de Gliomas mediante EfficientNet

Este proyecto tiene como objetivo el **entrenamiento de un modelo de clasificación de imágenes de gliomas** utilizando la arquitectura **EfficientNet**, un modelo de red neuronal convolucional preentrenado ampliamente utilizado en tareas de visión por computador.  
Imágenes extraidas del dataset REMBRANDT DEL TCIA: https://www.cancerimagingarchive.net/collection/rembrandt/
El trabajo se centra en la **clasificación automática de tumores cerebrales según su grado histológico (II, III y IV)** a partir de imágenes médicas, buscando aportar una herramienta que facilite la diferenciación de grados tumorales de forma automatizada y eficiente.

---

## 🧠 Descripción del proyecto

El código incluido en este repositorio implementa un flujo completo de **clasificación supervisada de imágenes**, que abarca las siguientes etapas principales:

1. **Carga y preprocesamiento de datos**  
   - Lectura y organización de las imágenes por grado tumoral (II, III, IV).  
   - Redimensionamiento, normalización y división en conjuntos de entrenamiento, validación y prueba.  

2. **Transfer learning con EfficientNet**  
   - Se emplea una arquitectura **EfficientNetB1** preentrenada en *ImageNet*, adaptada a la clasificación de tres clases.  
   - Las capas finales son modificadas para optimizar el rendimiento en este problema específico.  

3. **Entrenamiento y evaluación del modelo**  
   - Ajuste de hiperparámetros y optimización mediante técnicas de regularización y *batch normalization*.  
   - Evaluación de métricas de desempeño (accuracy, matriz de confusión, curvas ROC, etc.).  

4. **Guardado de resultados y modelo entrenado**  
   - El modelo final se guarda para su uso posterior en predicciones o validaciones externas.  

---

## 📂 Estructura del proyecto

├── Clasificación Glioma I_II_III_EfficientNet.ipynb # Notebook principal con el código
├── data/ # Carpeta con las imágenes organizadas por grado
│ ├── G_II/
│ ├── G_III/
│ └── G_IV/
├── models/ # Modelos entrenados (opcional)
└── README.md # Documento explicativo del proyecto
