Clasificación de Gliomas en Alto y Bajo Grado (LGG vs HGG)

Proyecto de *Deep Learning* orientado a la **clasificación automática de gliomas** (tumores cerebrales) en imágenes de **Resonancia Magnética (RM)**, empleando **Redes Neuronales Convolucionales (CNN)**.  

---

## 📂 Descripción del Proyecto

El objetivo principal de este proyecto es desarrollar un modelo capaz de **distinguir entre gliomas de bajo grado (LGG)** y **gliomas de alto grado (HGG)** a partir de imágenes de resonancia magnética cerebral.  

El trabajo se estructura en un flujo completo de *machine learning aplicado a imágenes médicas*, desde la carga y preprocesamiento de los datos hasta la evaluación final del modelo.

---

## ⚙️ Metodología

### 1. **Importación y Exploración de Datos**
- Lectura del conjunto de imágenes de RM con etiquetas LGG/HGG.
- Exploración básica del dataset (dimensiones, formatos, ejemplos visuales).
- División del dataset en subconjuntos de entrenamiento, validación y prueba.

### 2. **Preprocesamiento**
- Redimensionamiento de imágenes a un tamaño uniforme.
- Normalización de píxeles.
- Aumento de datos (*Data Augmentation*) para reducir sobreajuste: rotaciones, zoom, desplazamientos y flips horizontales/verticales.

### 3. **Construcción del Modelo**
- Implementación de una **Red Neuronal Convolucional (CNN)** con Keras/TensorFlow.
- Capas principales:
  - Convolución + ReLU
  - MaxPooling
  - Dropout
  - Capas densas finales con función *softmax* para clasificación binaria.
- Compilación del modelo con:
  - **Optimizador:** Adam  
  - **Función de pérdida:** Binary Crossentropy  
  - **Métricas:** Accuracy, Precision, Recall, F1-score, Especificidad.

### 4. **Entrenamiento**
- Entrenamiento supervisado sobre el conjunto de entrenamiento.
- Monitorización del rendimiento en el conjunto de validación.
- Ajuste de hiperparámetros: tasa de aprendizaje, número de épocas, tamaño de lote, número de filtros.

### 5. **Evaluación del Modelo**
- Evaluación sobre el conjunto de prueba.
- Visualización de métricas:
  - Matriz de confusión.
  - Curvas ROC y AUC.
  - Curvas de precisión-recall.
- Análisis de ejemplos correctamente e incorrectamente clasificados.
