# 🧠 Recorte Automático del Fondo Negro en Imágenes

## 📋 Descripción del código

Este script realiza el **recorte automático del fondo negro** en un conjunto de imágenes utilizando **TensorFlow**.  
El objetivo es eliminar las zonas vacías que rodean la región de interés, manteniendo únicamente el área con información útil.

El proceso que sigue el código es el siguiente:

1. **Lectura de las imágenes originales** desde una carpeta de entrada (`input_folder`).
2. **Conversión a escala de grises** para facilitar la detección de zonas oscuras.
3. **Creación de una máscara binaria** que distingue los píxeles negros (fondo) de los que contienen información.
4. **Cálculo de los límites de recorte**: se obtienen las coordenadas mínimas y máximas de los píxeles relevantes.
5. **Recorte automático de la imagen** para conservar solo el área informativa.
6. **Guardado del resultado** en una carpeta de salida (`output_folder`).

Si una imagen no contiene contenido (es completamente negra), el script la conserva sin cambios.

---

## 🎯 Interés y utilidad del cropping

El **cropping** o recorte de las regiones sin información es una etapa importante en el **preprocesamiento de imágenes biomédicas**.  
Sus beneficios principales son:

### 🪶 1. Reducción del tamaño de las imágenes  
Eliminar los márgenes negros disminuye la cantidad de píxeles procesados, lo que reduce el tamaño de los archivos y acelera la lectura y el entrenamiento.

### 🎯 2. Enfoque en la región relevante  
Las zonas oscuras no aportan información útil. Al recortar, el modelo se centra exclusivamente en la región con contenido (por ejemplo, un tumor o estructura anatómica).

### ⚙️ 3. Mejora del rendimiento del modelo  
El recorte elimina ruido visual y mejora la capacidad del modelo para aprender patrones verdaderamente significativos, lo que puede traducirse en mejores resultados en clasificación o segmentación.

### ⚡ 4. Ahorro computacional  
Menos píxeles implican menos operaciones de cálculo, lo que acelera el procesamiento y reduce el consumo de memoria, especialmente útil en equipos con recursos limitados.

### 📐 5. Normalización del dataset  
Las imágenes quedan más centradas y con proporciones consistentes, lo que favorece la homogeneidad del conjunto de datos y la estabilidad del entrenamiento.

---

## 🧩 En resumen

Este código automatiza una tarea clave del **preprocesamiento de imágenes**:  
> ✂️ **Eliminar el fondo negro para conservar únicamente la información relevante.**

Su uso permite optimizar la eficiencia, reducir el ruido visual y mejorar el rendimiento de los modelos de inteligencia artificial en análisis de imágenes médicas.

---
