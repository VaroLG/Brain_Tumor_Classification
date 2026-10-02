# Metodología: problemas del código original y cómo se corrigen

Este documento explica, paso a paso, qué cambió entre los notebooks del TFM
(`legacy/tfm_notebooks/`) y el paquete `btc`, y **por qué** cada cambio importa.
Está escrito para que se entienda sin haber leído el código.

---

## 1. Fuga de datos por paciente (el problema más grave)

**Qué pasaba.** De cada volumen 3D de RM se extraían muchos cortes 2D. Después,
todas las imágenes de una clase se barajaban (`random.shuffle`) y se repartían
60/15/25 en train/val/test.

**Por qué es un problema.** Dos cortes axiales contiguos del mismo paciente son
casi la misma imagen. Con el reparto por imagen, el modelo veía en
entrenamiento cortes vecinos de los pacientes de test. Puede "acertar"
reconociendo al paciente (su anatomía, su tumor concreto) en lugar de aprender
qué distingue a las clases. La exactitud de test deja de medir lo que importa:
cómo se comporta el modelo con **pacientes que no ha visto nunca**.

Las cifras del TFM lo hacen probable: en supervivencia de GBM había 1713
imágenes de 588 pacientes (≈3 cortes por paciente); en REMBRANDT, cientos de
imágenes de 109 pacientes.

**Corrección.** `btc.splits.patient_level_split` usa `StratifiedGroupKFold`:
todas las imágenes de un paciente caen en el mismo subconjunto y se mantiene la
proporción de clases. `assert_no_patient_leakage` se ejecuta tras el split y
otra vez antes de entrenar, así que un manifest con fuga no puede colarse.

Para que funcione, cada imagen necesita su `patient_id`:

- **UPENN-GBM:** el sujeto `UPENN-GBM-00001_11` (preoperatoria) y
  `UPENN-GBM-00001_21` (seguimiento) son el mismo paciente → `UPENN-GBM-00001`.
- **REMBRANDT:** se lee la etiqueta `PatientID` de la cabecera DICOM.
- **Kaggle:** no hay ID de paciente. Se agrupan las imágenes casi idénticas con
  un hash perceptual (dHash). Es una mitigación parcial (ver limitaciones).

## 2. Aumento de datos en validación y test

**Qué pasaba.** Un único `ImageDataGenerator` con rotaciones, desplazamientos,
zoom y volteos se usaba para crear los tres generadores (train, val y test).

**Por qué es un problema.** El aumento de datos es una técnica de
entrenamiento: genera variaciones para que el modelo no memorice. Aplicado en
test, cada evaluación usa imágenes deformadas al azar distintas, así que la
métrica cambia entre ejecuciones sin tocar el modelo, y además no se evalúa
sobre las imágenes reales.

**Corrección.** En `btc.datasets.make_dataset` el aumento solo se aplica con
`training=True`. Hay un test que comprueba que dos pasadas por el dataset de
evaluación dan exactamente los mismos tensores.

## 3. Preprocesado distinto al del preentrenamiento

**Qué pasaba.** El notebook de supervivencia normalizaba a [0, 1]
(`rescale=1./255`) y alimentaba VGG19. Los de grado y subtipo no aplicaban
ninguna normalización.

**Por qué es un problema.** Cada red de ImageNet se entrenó con una
normalización concreta y sus filtros esperan entradas en ese rango:

| Backbone | Entrada esperada |
|---|---|
| VGG19, ResNet50 | 0–255, BGR, restando la media de ImageNet ("caffe") |
| MobileNetV2, Xception | [-1, 1] |
| EfficientNet | 0–255 (normaliza internamente) |

Con VGG19 en [0, 1] las activaciones de las primeras capas son ~255 veces más
pequeñas de lo previsto y se pierde gran parte de lo que aporta el
preentrenamiento. (EfficientNet sin normalizar sí era correcto, por casualidad.)

**Corrección.** `btc.models.BACKBONES` asocia a cada red su `preprocess_input`,
y el pipeline lo aplica siempre, en train y en evaluación.

## 4. Supervivencia: media en lugar de mediana y censura ignorada

**Qué pasaba.** Los pacientes de GBM se partían en "SG alta" y "SG baja" según
superasen la media (18 meses).

**Por qué es un problema.**

- *Censura.* Si un paciente seguía vivo al cerrar el estudio, su "tiempo de
  supervivencia" es solo un mínimo. Un paciente vivo con 6 meses de
  seguimiento no tiene "SG baja": simplemente no sabemos aún. Etiquetarlo como
  tal mete errores en la variable que el modelo intenta aprender.
- *Media.* Unos pocos supervivientes muy largos desplazan la media hacia
  arriba; la mediana es robusta y da dos grupos del mismo tamaño.

**Corrección.** `btc.data.labels.survival_label`:

| Situación | Etiqueta |
|---|---|
| Falleció antes del umbral | `short` |
| Vivió (o se le siguió) al menos hasta el umbral | `long` |
| Censurado antes del umbral | excluido |

El paso siguiente correcto es no binarizar y tratarlo como supervivencia
(modelo de Cox, C-index), ver la hoja de ruta.

## 5. Selección manual de cortes

**Qué pasaba.** Se extraían todos los cortes (40 424 imágenes T1 axiales), se
descartaban los ficheros de menos de 10 KB y después se escogía a mano qué
imágenes mostraban el tumor.

**Por qué es un problema.** Es lento, subjetivo y no se puede repetir. Además
el tamaño del fichero PNG es un indicador indirecto de "corte con contenido".

**Corrección.** UPENN-GBM incluye segmentaciones del tumor.
`btc.data.nifti.select_slice_indices` se queda con los cortes cuya área tumoral
supera un 25 % de la del corte con más tumor y muestrea hasta 10 de forma
equiespaciada (no consecutivos, que serían casi duplicados). Por defecto usa
T1 con gadolinio, donde el realce del GBM es visible; el TFM usó T1 sin
contraste.

## 6. Cabeza sobredimensionada y épocas a mano

**Qué pasaba.** Sobre el backbone congelado se añadían
Dense(2560) → Dense(1280) → Dense(1024): unos 6 millones de parámetros nuevos
para unos cientos de pacientes. Las curvas del TFM muestran el resultado típico:
exactitud de train ≈ 0.95 y de validación ≈ 0.65–0.70, y 60 épocas daban peor
validación que 15.

**Corrección.**

- Cabeza `GlobalAveragePooling → Dropout → Dense(256) → Dropout → softmax`.
- `EarlyStopping` sobre la pérdida de validación con restauración de los
  mejores pesos.
- **Fine-tuning en dos fases**: primero solo la cabeza; después se descongelan
  las últimas capas del backbone con una tasa de aprendizaje 100 veces menor.
  La BatchNormalization del backbone se mantiene congelada. Si la segunda fase
  no mejora la validación se vuelve a los pesos de la primera.
- `class_weight` para compensar el desbalance (2:1 en supervivencia).

## 7. Evaluación por imagen y sin incertidumbre

**Qué pasaba.** Las métricas se calculaban por imagen y sin intervalo.

**Por qué es un problema.** La decisión clínica es por paciente, y un paciente
con 10 cortes pesaba 5 veces más que uno con 2. Además, con ~100 pacientes en
test, una diferencia de 0.03 entre modelos puede ser ruido.

**Corrección.** `btc.evaluate`:

- promedia las probabilidades de los cortes de cada paciente;
- reporta métricas por corte y por paciente (la de paciente es la que se cita);
- añade la **exactitud balanceada**, que no se deja engañar por el desbalance
  (predecir siempre "SG baja" daba 0.67 de exactitud);
- calcula IC 95 % por bootstrap remuestreando pacientes.

## 8. Reproducibilidad e ingeniería

| Antes | Ahora |
|---|---|
| Sin semillas | `set_seed` en Python, NumPy y TensorFlow |
| Rutas de Windows en el código | Rutas en argumentos de la CLI y en YAML |
| Mismas funciones copiadas 3 veces | Un módulo por responsabilidad |
| Imágenes copiadas a carpetas train/val/test | Un manifest CSV; el split es una columna |
| Clases por orden alfabético de carpetas ("Clase 0") | Lista `classes` explícita en la config |
| Un notebook que no ejecutaba (`Eff_model` sin definir, `Dense(2)` para 3 clases) | Tests de punta a punta en CI |
| `ImageDataGenerator` (obsoleto en Keras 3) | `tf.data` + capas de aumento de Keras |

## Limitaciones que siguen ahí

- **Kaggle sin ID de paciente.** El agrupamiento por dHash detecta duplicados
  reescalados o recomprimidos, pero no cortes distintos del mismo paciente.
- **REMBRANDT sin segmentaciones.** Los cortes se eligen por la franja central
  del volumen o con un CSV de selección manual.
- **Modelos 2D.** Se pierde la información tridimensional del tumor.
- **Un solo centro por tarea.** No hay validación externa (p. ej. entrenar en
  UPENN y probar en otra cohorte de GBM).
