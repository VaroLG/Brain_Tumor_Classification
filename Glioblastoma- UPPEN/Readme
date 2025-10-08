Extracción de Imágenes MRI de Glioblastomas (Dataset UPENN - TCIA)
📘 Descripción general

Este proyecto forma parte de mi trabajo final de master sobre la clasificación de imágenes de resonancia magnética nuclear (RMN) de tumores cerebrales mediante técnicas de Deep Learning.
El objetivo principal es procesar y extraer cortes bidimensionales (2D) de las imágenes MRI en formato NIfTI pertenecientes al dataset público de glioblastomas de la Universidad de Pensilvania (UPENN, disponible en TCIA), con el fin de preparar las imágenes para modelos de segmentación y clasificación.

🧩 Objetivos específicos

Cargar las imágenes de resonancia magnética en formato NIfTI (.nii/.gz).
Extraer las slices 2D relevantes de cada volumen tridimensional.
Guardar las imágenes resultantes en formato PNG, organizadas por:
- Paciente
- Modalidad (T1, T2, FLAIR, etc.)
- Orientación (axial, coronal, sagital)
- Asegurar una estructura clara y reutilizable para entrenamientos posteriores con modelos de IA.

Metodología y estructura del notebook:

1. Importación de librerías necesarias para la lectura, manejo y visualización de imágenes médicas y configuración del entorno y se cargan las librerías:

import nibabel as nib
import os
import matplotlib.pyplot as plt
import numpy as np


Estas librerías permiten manipular imágenes en formato NIfTI y realizar operaciones de conversión y guardado.

2. Definición de rutas de trabajo

Se establecen las rutas locales hacia los directorios donde se encuentran los archivos NIfTI descargados del dataset UPENN, así como el destino donde se guardarán las imágenes procesadas.

3. Lectura y exploración de los archivos NIfTI

En esta etapa se abre cada imagen NIfTI y se inspeccionan sus dimensiones para determinar el número de slices disponibles por orientación (axial, coronal, sagital).
Se utiliza nibabel.load() para leer cada archivo y get_fdata() para obtener la matriz de intensidades.

4. Extracción de cortes (slices) 2D

El código recorre cada volumen tridimensional y extrae imágenes 2D (slices).
Dependiendo de la orientación seleccionada (axial, coronal o sagital), se aplican distintos índices sobre el array de voxeles:

slice_img = volume[:, :, slice_index]


Cada slice se normaliza para mejorar la visualización y posteriormente se convierte a formato de imagen (PNG).

5. Visualización y verificación de cortes

Antes de guardar todas las imágenes, se visualizan algunos cortes con matplotlib para verificar que la extracción se ha realizado correctamente.
Esto permite comprobar la orientación, contraste y presencia de la lesión tumoral.

6. Guardado estructurado de imágenes en formato PNG

Cada imagen se guarda en carpetas organizadas por paciente, modalidad y orientación:

/Output/UPENN-GBM-00629/FLAIR/sagittal/slice_122.png

Los nombres de los archivos incluyen información relevante sobre el paciente, como el identificador del paciente, la modalidad de imagen, la orientación y el número de corte, lo cual permite enlazar fácilmente cada imagen con sus metadatos clínicos.
UPENN-GBM-00629_21_FLAIR_sagittal_slice_122.png

7. Automatización del proceso

El script está diseñado para procesar automáticamente todos los pacientes del conjunto, generando una base de datos visual completa y homogénea para etapas posteriores de entrenamiento con modelos de segmentación o clasificación.

🧠 Aplicaciones futuras

Entrenamiento de modelos de Deep Learning para segmentación tumoral.

Clasificación de pacientes según supervivencia o mutaciones moleculares.

Integración con datos clínicos o genómicos para análisis multimodal.
