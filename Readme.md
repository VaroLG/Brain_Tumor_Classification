# 🧠 Clasificación de Gliomas mediante Redes Neuronales Convolucionales (CNN)

Este repositorio recoge tres proyectos complementarios enfocados en la **clasificación automática de tumores cerebrales tipo glioma** a partir de imágenes médicas (MRI), empleando técnicas de **aprendizaje profundo (Deep Learning)** y **transfer learning**.  

El trabajo forma parte de un proyecto de investigación sobre el **análisis de resonancia magnética nuclear de tumores cerebrales con IA**, con el objetivo de desarrollar modelos predictivos que apoyen la práctica clínica en el diagnóstico y pronóstico de pacientes con glioma y glioblastoma.

---

## 🎯 Objetivo general

Desarrollar y evaluar modelos de clasificación basados en **redes neuronales convolucionales (CNN)** para distintos problemas clínicos asociados al glioma, aprovechando la información visual contenida en imágenes médicas de resonancia magnética anonimizadas.

---

## 📁 Estructura del proyecto

El repositorio se organiza en tres carpetas principales, cada una dedicada a un problema específico de clasificación.  

A continuación, se describe brevemente el contenido y propósito de cada módulo.

---

### 🧩 1. Clasificación del subtipo tumoral

Este módulo aborda dos problemas de clasificación:

- **Clasificación binaria:** distingue entre **gliomas de bajo grado (LGG)** y **gliomas de alto grado (HGG)**.  
- **Clasificación multiclase:** diferencia entre **gliomas de grado II, III y IV**.

El código organiza las imágenes por categorías y las divide automáticamente en conjuntos de **entrenamiento**, **validación** y **test** para su uso en modelos de aprendizaje profundo.

---

### 🧩 2. Clasificación del grado tumoral

Este módulo entrena un modelo basado en la arquitectura **EfficientNet**, aplicando *transfer learning* para clasificar las imágenes de gliomas en función de su grado histológico (II, III, IV).  

El enfoque combina la eficiencia computacional de EfficientNet con técnicas de **normalización y regularización**, logrando un modelo con buena capacidad de generalización y alta precisión.

---

### 🧩 3. Clasificación de la supervivencia global en glioblastomas

El tercer módulo busca predecir la **supervivencia global (SG)** de pacientes con **glioblastoma (GBM)** a partir de imágenes médicas.  

Las imágenes se clasifican en categorías de **alta y baja supervivencia**, empleando redes neuronales convolucionales adaptadas al tamaño y variabilidad del conjunto de datos.

---

## ⚙️ Tecnologías utilizadas

- **Lenguaje:** Python 3.9  
- **Entorno:** Jupyter Notebook / Anaconda  
- **Librerías principales:**
  - TensorFlow / Keras  
  - NumPy, Pandas, Matplotlib  
  - scikit-learn  

---

## 🧾 Conjuntos de datos

Los conjuntos de imágenes provienen de bases de datos públicas como **The Cancer Imaging Archive (TCIA)**, concretamente del **UPENN GBM Dataset**, en formato **NIfTI**, posteriormente procesado en **2D (slices)** y convertido a imágenes **PNG** para optimizar su manejo computacional.

---

## 🚀 Objetivos específicos

- Implementar una **pipeline reproducible** para la clasificación de imágenes médicas.  
- Evaluar modelos **preentrenados (EfficientNet, ResNet, etc.)** adaptados a distintos tipos de glioma.  
- Analizar la relación entre **imágenes radiológicas y variables clínicas** como el grado tumoral o la supervivencia.  

---

## 📈 Relevancia del proyecto

El uso de inteligencia artificial en neuroimagen abre la puerta a sistemas de apoyo al diagnóstico más **objetivos, rápidos y reproducibles**, que podrían ayudar a los especialistas a identificar patrones tumorales complejos y mejorar la toma de decisiones clínicas.

---

## 👨‍💻 Autor

**Álvaro Linacero Gracia**  
Proyecto desarrollado dentro del marco del **Trabajo Fin de Máster en Bioinformática y Bioestadística**, centrado en el análisis de imágenes de resonancia magnética de tumores cerebrales mediante técnicas de inteligencia artificial.

---

## 📜 Licencia

Este proyecto se distribuye bajo la licencia **MIT**, permitiendo su libre uso y modificación con fines académicos o de investigación.
