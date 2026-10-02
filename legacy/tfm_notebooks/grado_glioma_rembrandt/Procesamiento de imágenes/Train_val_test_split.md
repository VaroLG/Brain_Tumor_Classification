# 🧠 Clasificación de Imágenes de Glioma en Conjuntos de Entrenamiento, Validación y Test

## 📋 Descripción general

Este proyecto contiene código en Python para **organizar y preparar imágenes de gliomas** en conjuntos de **entrenamiento**, **validación** y **test**, con el objetivo de facilitar el entrenamiento de modelos de clasificación basados en imágenes médicas.

El código está implementado en formato Jupyter Notebook (`.ipynb`) y aborda **dos problemas de clasificación distintos**, ambos centrados en tumores cerebrales tipo glioma:

1. **Clasificación binaria** — Diferenciación entre **gliomas de alto grado (HGG)** y **gliomas de bajo grado (LGG)**.  
2. **Clasificación multiclase** — Categorización de los gliomas según su **grado histológico II, III o IV**.

Ambos casos emplean una lógica común para dividir automáticamente las imágenes disponibles en tres subconjuntos bien estructurados:
- **Entrenamiento (train)** → ~60% de las imágenes.  
- **Validación (val)** → ~15% de las imágenes.  
- **Prueba (test)** → ~25% de las imágenes.  

---

## ⚙️ Funcionamiento del código

El flujo general del script es el siguiente:

1. **Definición de directorios de origen**  
   Se especifican las carpetas que contienen las imágenes originales de cada categoría (por ejemplo, “HGG” y “LGG” o “G_II”, “G_III”, “G_IV”).

2. **Creación de la estructura de salida**  

Se generan automáticamente las carpetas de destino donde se guardarán las imágenes separadas por conjunto:
dataset/
│
├── train/
│   ├── clase_1/
│   ├── clase_2/
│
├── val/
│   ├── clase_1/
│   ├── clase_2/
│
└── test/
    ├── clase_1/
    ├── clase_2/
