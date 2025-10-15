# 🧠 Clasificación de Subtipos tumorales de cerebro con EfficientNetB1

Este proyecto implementa una red neuronal convolucional basada en **EfficientNetB1** para la **clasificación automática de subtipos tumorales de cerebro** a partir de imágenes de resonancia magnética (MRI) de un dataset de Kaggle.  

https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

Se utiliza un enfoque de **transfer learning**, ajustando las capas superiores del modelo preentrenado e incorporando capas densas personalizadas con técnicas de regularización y normalización.

---

## ⚙️ Descripción del Proyecto

El objetivo es clasificar imágenes de gliomas según su **subtipo tumoral** (Glioma, Meningioma, Tumor de la glándula hipófisis y ausencia de tumor), optimizando la precisión del modelo mediante **entrenamiento por etapas** y estrategias de **regularización** como *Dropout*.

El flujo general del proyecto incluye:
1. Carga y preprocesamiento de datos.
2. Construcción y configuración de la red EfficientNetB1.
3. Entrenamiento del modelo en múltiples etapas.
4. Evaluación mediante métricas de rendimiento y visualización de resultados.

---

## 🧩 Arquitectura del Modelo

El modelo se basa en la arquitectura **EfficientNetB1**, que optimiza simultáneamente profundidad, ancho y resolución de la red para obtener un equilibrio entre **rendimiento y eficiencia computacional**.  

A partir de esta base preentrenada (en ImageNet), se añaden capas densas personalizadas para adaptar el modelo a la tarea de clasificación de subtipos de tumor.

### 🔹 Estructura general:

1. **Base convolucional (EfficientNetB1 preentrenada)**  
   - Se utiliza como *feature extractor*, congelando inicialmente sus pesos para aprovechar el conocimiento previo aprendido sobre patrones visuales.  
   - Incluye bloques MBConv (Mobile Inverted Bottleneck Convolution) con conexiones residuales, *batch normalization* y activaciones *Swish*.

2. **Global Average Pooling (GAP)**  
   - Reduce el volumen tridimensional de salida de la EfficientNet a un vector 1D, resumiendo la información espacial de cada mapa de activación.  
   - Evita el uso de capas Flatten, mejorando la eficiencia y reduciendo el sobreajuste.

3. **Capas Densas Personalizadas**
   - Se añaden varias capas **Dense** (totalmente conectadas) con activación ReLU.  
   - Estas capas permiten combinar las características extraídas por la base convolucional y generar representaciones más específicas para la tarea de clasificación.

4. **Batch Normalization**
   - Normaliza las activaciones de cada capa, estabilizando el entrenamiento y acelerando la convergencia.  
   - Ayuda a reducir la sensibilidad a los cambios en la distribución de entrada.

5. **Dropout**
   - Se utiliza tras las capas densas para evitar el sobreajuste.  
   - Durante el entrenamiento, desactiva aleatoriamente una fracción de neuronas (por ejemplo, `rate=0.5`), lo que obliga al modelo a no depender de conexiones específicas.  
   - En inferencia, todas las neuronas están activas, pero sus salidas se escalan proporcionalmente a la tasa de Dropout.  
   - Resultado: un modelo más robusto y generalizable.

6. **Capa de salida (Softmax)**
   - Número de neuronas igual al número de clases (en este caso, 4).  
   - Función de activación **Softmax**, que convierte las salidas en probabilidades normalizadas para cada clase.

---

## 🧠 Entrenamiento por Etapas

El modelo se entrena en **30 etapas** para lograr una convergencia estable:

1. **Primera etapa:**  
   - Se entrena únicamente la parte superior de la red (capas densas personalizadas).  
   - La base EfficientNetB1 permanece congelada.

2. **Etapas posteriores:**  
   - Se descongelan progresivamente capas de la EfficientNetB1 para realizar *fine-tuning*.  
   - Se aplican tasas de aprendizaje más pequeñas para no alterar los pesos preentrenados de forma drástica.

3. **Optimizador y funciones de pérdida:**  
   - Optimizador: **Adam** o **AdamW**.  
   - Función de pérdida: **Categorical Crossentropy**.  
   - Métricas de evaluación: *Accuracy, Recall, Precision, F1-score* y *Especificidad*.

---

## 📊 Evaluación del Modelo

Durante el entrenamiento y validación se monitorizan las siguientes métricas:

- **Accuracy (exactitud global)**  
- **Precision y Recall por clase**  
- **F1-score promedio**  
- **Especificidad (minimiza falsos positivos)**  

Además, se generan las siguientes visualizaciones:
- Curvas de pérdida y precisión por época.  
- Matriz de confusión.  
- Curva ROC y área bajo la curva (AUC).

---

## 🧱 Estructura del Repositorio


