# Hoja de ruta: de la clasificación por imagen a la radiogenómica

La conclusión del TFM fue que predecir supervivencia o mutaciones solo con
cortes 2D de RM es muy difícil, y proponía como trabajo futuro combinar la
imagen con datos clínicos y ómicos. Este documento ordena ese trabajo futuro.

## Por qué la ómica es la extensión natural

Desde la clasificación de la OMS de 2021 (CNS5), los gliomas se definen con
marcadores moleculares además de la histología: el estado de **IDH**, la
**codeleción 1p/19q**, y en el glioblastoma IDH-wildtype alteraciones como la
amplificación de **EGFR**, mutaciones del promotor de **TERT** o la ganancia del
cromosoma 7 con pérdida del 10. La metilación del promotor de **MGMT** predice
la respuesta a temozolomida. Un "grado" predicho solo desde la imagen se queda
corto frente a esa definición; conectar imagen y genoma es lo que da sentido
clínico al modelo.

## Fase 0 — Reentrenar con la metodología corregida

- [ ] Reconstruir los manifests de las cuatro tareas del TFM con `btc manifest-*`.
- [ ] Entrenar con `configs/*.yaml` y publicar en el README las métricas por
      paciente con IC 95 %, junto a las históricas.
- [ ] Comparar backbones (EfficientNetB0/B1, VGG19, ResNet50) con el mismo split.

## Fase 1 — Radiogenómica con lo que ya tiene UPENN-GBM

UPENN-GBM ya incluye, además de las RM y segmentaciones, el estado de IDH1 y de
MGMT y features radiómicas precalculadas. Sin descargar nada nuevo:

- [ ] `btc manifest-upenn --task idh` y `--task mgmt` (ya implementado).
- [ ] Modelo tabular con radiómica + edad + sexo (regresión logística, gradient
      boosting) como *baseline* fuerte frente a la CNN.
- [ ] Fusión tardía: embedding de la CNN + variables clínicas/moleculares.
- [ ] **Supervivencia como supervivencia**: modelo de Cox (`lifelines` o
      `scikit-survival`) con C-index, comparando imagen sola frente a
      imagen + IDH + MGMT + edad. Aprovecha los pacientes censurados que la
      binarización actual tiene que descartar.

## ✅ Implementado: necrosis en RM ↔ hipoxia multiómica

El estudio concreto elegido para demostrar la línea radiogenómica está
implementado en `src/btc/radiogenomics/` con plan prerregistrado en
[`PLAN_ANALISIS_RADIOGENOMICA.md`](PLAN_ANALISIS_RADIOGENOMICA.md):
descubrimiento en TCGA-GBM (microarray U133A), validación en CPTAC-GBM (ARN y
proteína) y exploración de todas las capas ómicas. Pendiente: ejecutarlo con
los datos reales tras comprobar el solapamiento (`btc rg-overlap`).

## Fase 2 — Imagen + genoma con TCGA

Los pacientes de TCGA-GBM y TCGA-LGG tienen RM en TCIA y, por otro lado,
mutaciones, expresión génica y clínica en TCGA (cBioPortal / GDC). Hay
segmentaciones y features radiómicas publicadas para sus RM preoperatorias
(Bakas et al., *Sci Data* 2017; 135 GBM + 108 LGG).

- [ ] Predecir IDH y 1p/19q desde la RM con verdad de referencia genómica.
- [ ] Puntuar firmas transcriptómicas (p. ej. subtipos proneural / clásico /
      mesenquimal) y estudiar su relación con lo que ve el modelo.

## Fase 3 — Hábitats tumorales: segmentación ↔ transcriptómica espacial

Conecta con la línea de segmentación (YOLO). El Ivy Glioblastoma Atlas Project
tiene RNA-seq de estructuras anatómicas microdiseccionadas del GBM (borde de
avance, tumor infiltrante, tumor celular, proliferación microvascular y
pseudoempalizada alrededor de la necrosis) y RM de su cohorte en TCIA. Esas
estructuras se corresponden aproximadamente con las clases que se segmentan:

| Subregión segmentada (BraTS) | Estructura Ivy GAP | Biología esperada |
|---|---|---|
| Necrosis | Pseudoempalizada / necrosis | Hipoxia, estrés |
| Realce | Tumor celular + proliferación microvascular | Proliferación, angiogénesis |
| Edema | Tumor infiltrante / borde de avance | Invasión |

- [ ] Expresión diferencial entre estructuras (Ivy GAP) → firmas por hábitat.
- [ ] Puntuar esas firmas en TCGA-GBM y correlacionarlas con la fracción de
      necrosis / realce / edema medida en la segmentación.
- [ ] ¿Aporta la composición de hábitats valor pronóstico más allá de IDH y MGMT?

## Fase 4 — Ingeniería

- [ ] Modelos 3D o 2.5D (varios cortes como canales).
- [ ] Validación externa entre cohortes (UPENN ↔ TCGA ↔ CPTAC).
- [ ] Explicabilidad (Grad-CAM) para comprobar que el modelo mira el tumor.
- [ ] Exportación a ONNX/TensorRT para inferencia en el borde (Jetson).

## Referencias de datos

- UPENN-GBM — Bakas et al., *Sci Data* 2022; <https://www.cancerimagingarchive.net/collection/upenn-gbm/>
- Segmentaciones TCGA-GBM/LGG — Bakas et al., *Sci Data* 2017; <https://doi.org/10.1038/sdata.2017.117>
- BraTS 2021 (segmentación + MGMT) — Baid et al., arXiv:2107.02314
- Ivy GAP — <https://glioblastoma.alleninstitute.org/>; RM en TCIA <https://doi.org/10.7937/K9/TCIA.2016.XLWAN6NL>
